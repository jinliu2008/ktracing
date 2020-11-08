import os
import math
import copy
import torch
import json
import time
import random
import logging
import argparse
import collections
import numpy as np
import pandas as pd
import ktracing_data
import ktracing_models
import pytorch_lightning.metrics.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn import metrics
import warnings
from ktracing_utils import *
import sys

warnings.filterwarnings(action='ignore')

settings = json.load(open('SETTINGS.json'))
parameters = read_yml('parameters.yaml')

DB_PATH = settings['CLEAN_DATA_DIR']
FINETUNED_MODEL_PATH = settings['MODEL_DIR']
time_str = time.strftime("%m%d-%H%M")

# get_logger(settings, time_str)

logging.basicConfig(filename=os.path.join(settings['LOGS_DIR'], f'log_{time_str}.txt'),
                    level=logging.INFO, format="%(message)s")

class CFG:
    learning_rate = 3.0e-3
    batch_size = 128
    num_workers = 4
    print_freq = 100
    test_freq = 1
    start_epoch = 0
    num_train_epochs = 5
    warmup_steps = 1
    max_grad_norm = 100
    gradient_accumulation_steps = 1
    weight_decay = 0.01
    dropout = 0.2
    emb_size = 50
    hidden_size = 10
    nlayers = 2
    nheads = 10
    seq_len = 50


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--data", type=str, default='bowl.pt')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--use_test", action='store_true')
    parser.add_argument("--aug", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--grad_accums", type=int, default=CFG.gradient_accumulation_steps)
    parser.add_argument("--nepochs", type=int, default=CFG.num_train_epochs)
    parser.add_argument("--wsteps", type=int, default=CFG.warmup_steps)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data_seed", type=int, default=7)
    parser.add_argument("--seq_len", type=int, default=CFG.seq_len)
    parser.add_argument("--nlayers", type=int, default=CFG.nlayers)
    parser.add_argument("--nheads", type=int, default=CFG.nheads)
    parser.add_argument("--hidden_size", type=int, default=CFG.hidden_size)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--lr", type=float, default=CFG.learning_rate)
    parser.add_argument("--dropout", type=float, default=CFG.dropout)
    parser.add_argument("--encoder", type=str, default='TRANSFORMER')
    # parser.add_argument("--encoder", type=str, default='LSTM')
    args = parser.parse_args()



    CFG.batch_size = args.batch_size
    CFG.gradient_accumulation_steps = args.grad_accums
    CFG.batch_size = CFG.batch_size // CFG.gradient_accumulation_steps
    CFG.num_train_epochs = args.nepochs
    CFG.warmup_steps = args.wsteps
    CFG.learning_rate = args.lr
    CFG.dropout = args.dropout
    CFG.seed = args.seed
    CFG.data_seed = args.data_seed
    CFG.seq_len = args.seq_len
    CFG.nlayers = args.nlayers
    CFG.nheads = args.nheads
    CFG.hidden_size = args.hidden_size
    CFG.res_dir = f'res_dir_{args.k}'
    CFG.target_size = 1
    CFG.encoder = args.encoder
    CFG.aug = args.aug

    os.environ['PYTHONHASHSEED'] = str(CFG.seed)
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)
    torch.cuda.manual_seed(CFG.seed)
    torch.backends.cudnn.deterministic = True

    (train_df, train_samples, cate_offset) = torch.load(os.path.join(settings['CLEAN_DATA_DIR'], settings['TRAIN_PT']))
    cont_cols = parameters['cont_cols']
    cate_cols = parameters['cate_cols']

    print('shape: ', train_df.shape)
    print(cate_cols, cont_cols)

    CFG.total_cate_size = cate_offset
    CFG.cate_cols = cate_cols
    CFG.cont_cols = cont_cols

    model = ktracing_models.encoders[CFG.encoder](CFG)
    model.cuda()
    model._dropout = CFG.dropout

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info(f'CFG: {CFG.__dict__}')
    logging.info(f'arg: {args}')
    logging.info(f'parameters: {count_parameters(model)}')

    train_db = ktracing_data.KTDataset(CFG, train_df, train_samples, aug=CFG.aug)

    train_loader = DataLoader(
        train_db, batch_size=CFG.batch_size, shuffle=True,
        num_workers=CFG.num_workers, pin_memory=True)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=CFG.learning_rate,
                      weight_decay=CFG.weight_decay,
                      )

    num_train_optimization_steps = int(
        len(train_db) / CFG.batch_size / CFG.gradient_accumulation_steps) * 7
    logging.info(f'num_train_optimization_steps: {num_train_optimization_steps}')
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=CFG.warmup_steps,
                                                num_training_steps=num_train_optimization_steps
                                                )

    logging.info('use WarmupLinearSchedule ...')

    def get_lr():
        return scheduler.get_lr()[0]

    log_df = pd.DataFrame(columns=(['EPOCH'] + ['LR'] + ['TRAIN_LOSS', 'auc']))

    # os.makedirs('log', exist_ok=True)
    # get_logger(settings)

    CFG.input_filename = settings['TRAIN_PT']
    file_name = generate_file_name(CFG)
    curr_lr = get_lr()
    logging.info(f'using training data: {file_name}')
    logging.info(f'initial learning rate:{curr_lr}')

    best_model = None
    best_epoch = 0

    model_list = []

    for epoch in range(CFG.start_epoch, CFG.num_train_epochs):
        # train for one epoch

        train_loss, auc = train(train_loader, model, optimizer, epoch, scheduler)

        if epoch % CFG.test_freq == 0 and epoch >= 0:
            log_row = {'EPOCH': epoch, 'LR': curr_lr, 'TRAIN_LOSS': train_loss, 'auc': auc}
            log_df = log_df.append(pd.DataFrame(log_row, index=[0]), sort=False)
            logging.info(log_df.tail(20))

            batch_size = CFG.batch_size * CFG.gradient_accumulation_steps

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the cust_model it-self

    curr_model_name = f"{file_name}_{args.k}.pt"
    # curr_model_name = (f'b-{batch_size}_a-{CFG.encoder}_e-{CFG.emb_size}_h-{CFG.hidden_size}_'
    #                    f'd-{CFG.dropout}_l-{CFG.nlayers}_hd-{CFG.nheads}_'
    #                    f's-{CFG.seed}_len-{CFG.seq_len}_aug-{CFG.aug}_da-{input_filename}_k-{args.k}.pt')

    save_checkpoint({
        'epoch': best_epoch + 1,
        'arch': 'transformer',
        'state_dict': model_to_save.state_dict(),
        'log': log_df,
    },
        FINETUNED_MODEL_PATH, curr_model_name,
    )
    print('done')


def train(train_loader, model, optimizer, epoch, scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    sent_count = AverageMeter()

    # switch to train mode
    model.train()
    start = time.time()
    end = time.time()
    global_step = 0

    for step, (cate_x, cont_x, mask, y) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        cate_x, cont_x, mask, y = cate_x.cuda(), cont_x.cuda(), mask.cuda(), y.cuda()
        batch_size = cate_x.size(0)

        # compute loss
        pred = model(cate_x, cont_x, mask)
        loss = torch.nn.BCELoss()(pred, y)

        # record loss
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
        else:
            loss.backward()

        # measure elapsed time
        batch_time.update(time.time() - end)
        sent_count.update(batch_size)
        end = time.time()

        try:
            auc = metrics.roc_auc_score(y.detach().cpu().numpy(), pred.detach().cpu().numpy())
            accuracies.update(auc, batch_size)
        except:
            logging.error('exception occured. current averaged auc:', accuracies.avg)

        if (step % 200) == 0:
            logging.info(f'curr loss: {auc}')
            logging.info('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Acc: {acc.val:.4f}({acc.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.6f}  '
                  'sent/s {sent_s:.0f} '
                .format(
                epoch, step, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses,
                acc=accuracies,
                remain=timeSince(start, float(step + 1) / len(train_loader)),
                grad_norm=grad_norm,
                lr=scheduler.get_lr()[0],
                sent_s=sent_count.avg / batch_time.avg
            ))

    print('overall losses:', losses.avg)
    print('overall auc:', accuracies.avg)
    return losses.avg, accuracies.avg


def save_checkpoint(state, model_path, model_filename, is_best=False):
    print('saving cust_model ...')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(state, os.path.join(model_path, model_filename))
    if is_best:
        torch.save(state, os.path.join(model_path, 'best_' + model_filename))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        # print('update curr:', val, 'sum:', self.sum, self.count)
        # print('average: ', self.avg)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def adjust_learning_rate(optimizer, epoch):
    # lr  = CFG.learning_rate
    lr = (CFG.lr_decay) ** (epoch // 10) * CFG.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    main()

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

import warnings

warnings.filterwarnings(action='ignore')

settings = json.load(open('SETTINGS.json'))
DB_PATH = settings['CLEAN_DATA_DIR']
FINETUNED_MODEL_PATH = settings['MODEL_DIR']


class CFG:
    learning_rate = 1.0e-4
    batch_size = 256
    num_workers = 4
    print_freq = 100
    test_freq = 1
    start_epoch = 0
    num_train_epochs = 1
    warmup_steps = 30
    max_grad_norm = 1000
    gradient_accumulation_steps = 1
    weight_decay = 0.01
    dropout = 0.2
    emb_size = 100
    hidden_size = 500
    nlayers = 2
    nheads = 10
    seq_len = 2


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
    # parser.add_argument("--encoder", type=str, default='TRANSFORMER')
    parser.add_argument("--encoder", type=str, default='LSTM')
    args = parser.parse_args()
    print(args)

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
    print(CFG.__dict__)

    os.environ['PYTHONHASHSEED'] = str(CFG.seed)
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)
    torch.cuda.manual_seed(CFG.seed)
    torch.backends.cudnn.deterministic = True

    data_path = "ktracing_train.pt"
    (train_samples, train_users, train_df, mappers_dict, cate_offset, cate_cols, cont_cols) = (
        torch.load(data_path))
    print(data_path)
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

    print('parameters: ', count_parameters(model))

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
    print('num_train_optimization_steps', num_train_optimization_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=CFG.warmup_steps,
                                                num_training_steps=num_train_optimization_steps
                                                )

    print('use WarmupLinearSchedule ...')

    def get_lr():
        return scheduler.get_lr()[0]

    log_df = pd.DataFrame(columns=(['EPOCH'] + ['LR'] + ['TRAIN_LOSS', 'auc']))

    os.makedirs('log', exist_ok=True)

    curr_lr = get_lr()

    print(f'initial learning rate:{curr_lr}')

    best_model = None
    best_epoch = 0

    model_list = []

    for epoch in range(CFG.start_epoch, CFG.num_train_epochs):
        # train for one epoch

        train_loss = train(train_loader, model, optimizer, epoch, scheduler)

        if epoch % CFG.test_freq == 0 and epoch >= 0:
            log_row = {'EPOCH': epoch, 'LR': curr_lr,
                       'TRAIN_LOSS': train_loss
                       }
            log_df = log_df.append(pd.DataFrame(log_row, index=[0]), sort=False)
            print(log_df.tail(20))

            batch_size = CFG.batch_size * CFG.gradient_accumulation_steps

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the cust_model it-self

    input_filename = args.data.split('/')[-1]
    curr_model_name = (f'b-{batch_size}_a-{CFG.encoder}_e-{CFG.emb_size}_h-{CFG.hidden_size}_'
                       f'd-{CFG.dropout}_l-{CFG.nlayers}_hd-{CFG.nheads}_'
                       f's-{CFG.seed}_len-{CFG.seq_len}_aug-{CFG.aug}_da-{input_filename}_k-{args.k}.pt')
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
    # meter = bowl_utils.Meter()

    # switch to train mode
    model.train()

    start = end = time.time()
    global_step = 0

    for step, (cate_x, cont_x, mask, y) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        cate_x, cont_x, mask, y = cate_x.cuda(), cont_x.cuda(), mask.cuda(), y.cuda()
        batch_size = cate_x.size(0)

        # compute loss
        k = 0.5
        # y = y[:,-1,:]#
        pred = model(cate_x, cont_x, mask)
        try:
            loss = torch.nn.BCELoss()(pred, y)
        except:
            print('curr loss: ', F.classification.auc(pred.view(-1), y.view(-1)))
        # record loss

        #losses.update(loss.item(), batch_size)
        print('current loss:', loss.item())
        losses.update(loss.item(), 1)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        sent_count.update(batch_size)
    return losses.avg


def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger


logger = get_logger()


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

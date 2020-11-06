import os
import sys

os.environ['OMP_NUM_THREADS'] = '16'
os.environ['NUMEXPR_MAX_THREADS'] = '16'
import os
import math
import copy
import torch
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from transformers import AdamW, get_linear_schedule_with_warmup
import ktracing_data
import ktracing_models
import pytorch_lightning.metrics.functional as F
from torch.utils.data import DataLoader
import warnings
import ktracing_models
warnings.filterwarnings(action='ignore')

import argparse
import logging
import json
import collections

settings = json.load(open('SETTINGS.json'))

DB_PATH = settings['CLEAN_DATA_DIR']
FINETUNED_MODEL_PATH = settings['MODEL_DIR']

class CFG:
    learning_rate = 1.0e-3
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
    emb_size = 100
    hidden_size = 20
    nlayers = 2
    nheads = 10
    seq_len = 100

def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--nepochs", type=int, default=CFG.num_train_epochs)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data_seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=CFG.learning_rate)
    parser.add_argument("--dropout", type=float, default=CFG.dropout)
    args = parser.parse_args()
    print(args)

    CFG.batch_size = args.batch_size
    CFG.seed = args.seed
    CFG.data_seed = args.data_seed
    CFG.target_size = 1
    print(CFG.__dict__)

    os.environ['PYTHONHASHSEED'] = str(CFG.seed)
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)
    torch.cuda.manual_seed(CFG.seed)
    torch.backends.cudnn.deterministic = True


    # data_path = "ktracing_train_v0.pt"
    data_path = "ktracing_train_v1.pt"
    (valid_samples, valid_users, valid_df, mappers_dict, cate_offset, cate_cols, cont_cols) = (
        torch.load(data_path))

    cont_cols = ['prior_question_elapsed_time', 'lagged_time', "answered_correctly_content"]
    print(data_path)
    print('shape: ', valid_df.shape)
    print(cate_cols, cont_cols)


    CFG.total_cate_size = cate_offset
    CFG.cate_cols = cate_cols
    CFG.cont_cols = cont_cols

    path = 'b-128_a-TRANSFORMER_e-100_h-20_d-0.2_l-2_hd-10_s-7_len-100_aug-0.0_da-bowl.pt_k-0.pt'
    cfg_dict = dict([tok.split('-') for tok in path.replace('bowl_', '').split('_')])
    CFG.encoder = cfg_dict['a']
    CFG.seq_len = int(cfg_dict['len'])
    CFG.emb_size = int(cfg_dict['e'])
    CFG.hidden_size = int(cfg_dict['h'])
    CFG.nlayers = int(cfg_dict['l'])
    CFG.nheads = int(cfg_dict['hd'])

    model_path = os.path.join(settings['MODEL_DIR'], path)
    model = ktracing_models.encoders[CFG.encoder](CFG)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))
    model.cuda()

    predictions = 0
    valid_db = ktracing_data.KTDataset(CFG, valid_df, valid_samples)
    valid_loader = DataLoader(
        valid_db, batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=True)
    auc_score, prediction, groundtruth = validate(valid_loader, model)

    auc = metrics.roc_auc_score(groundtruth,prediction)
    print('current loss: ', auc)


def validate(valid_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    sent_count = AverageMeter()
    # meter = bowl_utils.Meter()

    # switch to evaluation mode
    model.eval()

    start = end = time.time()

    predictions = []
    groundtruth = []
    for step, (cate_x, cont_x, mask, y) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        cate_x, cont_x, mask, y = cate_x.cuda(), cont_x.cuda(), mask.cuda(), y.cuda()
        batch_size = cate_x.size(0)

        # compute loss
        k = 0.5
        with torch.no_grad():
            pred = model(cate_x, cont_x, mask)
            
            loss = metrics.roc_auc_score(y.detach().cpu().numpy(), pred.detach().cpu().numpy())

            # record loss
            losses.update(loss.item(), batch_size)

        pred_y = pred.detach().cpu()
        y = y.cpu()
        predictions.append(pred_y)
        groundtruth.append(y)

    predictions = torch.cat(predictions).numpy()
    groundtruth = torch.cat(groundtruth).numpy()

    return losses.avg, predictions, groundtruth


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

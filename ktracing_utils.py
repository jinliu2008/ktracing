import os
import yaml

import pandas as pd
import gcsfs
import feather
from sklearn import metrics
import sqlite3
import gc

from tqdm import tqdm as tqdm_notebook

from collections import deque
import torch

import pickle
import _pickle as cpickle
import logging
import math
import numpy as np
import time
from ktracing_data import *
from ktracing_models import *
from torch.utils.data import DataLoader


############################################################
# Configuration Utility
############################################################


dtypes = {
    "row_id": "int64",
    "timestamp": "int64",
    "user_id": "int32",
    "content_id": "int16",
    "content_type_id": "boolean",
    "task_container_id": "int16",
    "user_answer": "int8",
    "answered_correctly": "int8",
    "prior_question_elapsed_time": "float32",
    "prior_question_had_explanation": "boolean"
}


class CFG:
    learning_rate = 1.0e-3
    batch_size = 128
    num_workers = 8
    print_freq = 100
    test_freq = 1
    start_epoch = 0
    num_train_epochs = 10
    warmup_steps = 1
    max_grad_norm = 100
    gradient_accumulation_steps = 1
    weight_decay = 0.01
    dropout = 0.2
    emb_size = 50
    hidden_size = 20
    nlayers = 1
    nheads = 10
    seq_len = 50
    target_size = 1
    res_dir = 'res_dir_0'
    seed = 123
    data_seed = 123
    encoder = 'TRANSFORMER'
    aug = 0.0
    window_size = 50


def get_logger(settings, time_str):
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(filename=os.path.join(settings['LOGS_DIR'], f'log_{time_str}.txt'),
                        format=FORMAT, level=logging.INFO)
    logger = logging.getLogger('main')
    return logger


def read_yml(yml_path):
    yml_path = os.path.normpath(yml_path)
    with open(yml_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def generate_file_name(CFG):
    batch_size = CFG.batch_size * CFG.gradient_accumulation_steps
    input_filename = CFG.input_filename.split('.')[0].replace('_','').replace('-','')
    file_name = f'b-{batch_size}_a-{CFG.encoder}_e-{CFG.emb_size}_h-{CFG.hidden_size}_d' \
                f'-{CFG.dropout}_l-{CFG.nlayers}_hd-{CFG.nheads}_s-{CFG.seed}_len-{CFG.seq_len}_aug' \
                f'-{CFG.aug}_da-{input_filename}'
    return file_name


def get_questions_df(settings):
    questions_df = pd.read_csv(os.path.join(settings['RAW_DATA_DIR'], 'questions.csv'), usecols=[0, 3],
                               dtype={'question_id': 'int16', 'part': 'int8'})
    questions_df.rename(columns={'question_id': 'content_id'}, inplace=True)
    questions_df.set_index('content_id', inplace=True)
    return questions_df

def get_lectures_df(settings):
    lectures_df = pd.read_csv(os.path.join(settings['RAW_DATA_DIR'], 'lectures.csv'), usecols=[0, 1, 3])
    lectures_df.rename(columns={'lecture_id': 'content_id'}, inplace=True)
    lectures_df.set_index('content_id', inplace=True)
    return lectures_df


def generate_files(settings, CFG, submission=False):
    if submission:
        input_file_name = 'train.feather'
        file_path = settings["SUBMISSION_DIR"]
    else:
        input_file_name = 'train_v0.feather'
        file_path = settings["CLEAN_DATA_DIR"]

    cate_cols = CFG.cate_cols

    mappers_dict_path = os.path.join(file_path, 'mappers_dict.pkl')
    questions_df = get_questions_df(settings)
    lectures_df = get_lectures_df(settings)
    if not os.path.isfile(mappers_dict_path):
        df_ = feather.read_dataframe(os.path.join(settings["RAW_DATA_DIR"], 'train.feather'))

        lectures_df = get_lectures_df(settings)
        df_.set_index('content_id', inplace=True)
        df_ = df_.join(questions_df, how='left')
        df_['part'].fillna(value=-1, inplace=True)
        df_ = df_.join(lectures_df, how='left')
        df_['type_of'].fillna(value='None', inplace=True)
        df_.reset_index(inplace=True)
        mappers_dict = {}
        cate_offset = 1
        for col in (cate_cols):
            cate2idx = {}
            for v in df_[col].unique():
                if (v != v) |pd.isna(v)| (v == None): continue
                cate2idx[v] = len(cate2idx) + cate_offset
            mappers_dict[col] = cate2idx
            cate_offset += len(cate2idx)
        with open(mappers_dict_path, 'wb') as handle:
            pickle.dump(mappers_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(mappers_dict_path, 'rb') as handle:
            mappers_dict = pickle.load(handle)

    results_c_path = os.path.join(file_path, 'results_c.pkl')
    if not os.path.isfile(results_c_path):
        df_ = feather.read_dataframe(os.path.join(settings["RAW_DATA_DIR"], input_file_name))
        results_c = df_[df_.content_type_id == 0].groupby(['content_id'])['answered_correctly'].agg(['mean'])
        results_c.columns = ["answered_correctly_content"]
        with open(results_c_path, 'wb') as handle:
            pickle.dump(results_c, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(results_c_path, 'rb') as handle:
            results_c = pickle.load(handle)

    return questions_df, lectures_df, mappers_dict, results_c



def build_conn(df_users_content, chunk_size=20000):
    conn = sqlite3.connect(':memory:')
    # cursor = conn.cursor()

    total = len(df_users_content)
    n_chunks = (total // chunk_size + 1)

    i = 0
    while i < n_chunks:
        df_users_content.iloc[i * chunk_size:(i + 1) * chunk_size].to_sql('results_u', conn, method='multi',
                                                                          if_exists='append', index=False)
        i += 1

    conn.execute('CREATE UNIQUE INDEX users_index ON results_u user_id')
    del df_users_content
    gc.collect()
    return conn


def get_user_dict(settings, CFG, submission_flag=True):

    if submission_flag:
        results_u_path = os.path.join(settings["SUBMISSION_DIR"], 'user_dict.pkl')
    else:
        results_u_path = os.path.join(settings["CLEAN_DATA_DIR"], 'user_dict.pkl')

    if not os.path.isfile(results_u_path):
        if submission_flag:
            input_file_name = "train.feather"
        else:
            input_file_name = "train_v0.feather"

        df_ = feather.read_dataframe(os.path.join(settings['RAW_DATA_DIR'], input_file_name))
        print('columns:', df_.columns.to_list())

        df_.sort_values(['timestamp'], ascending=True, inplace=True)
        df_.reset_index(drop=True, inplace=True)

        df_ = df_.groupby('user_id').tail(CFG.window_size)
        df_, _, _ = preprocess_data(df_, settings, CFG)
        df_ = df_[['user_id'] + CFG.features]
        user_dict = {uid: u.values[:, 1:] for uid, u in df_.groupby('user_id')}
        with open(results_u_path, 'wb') as handle:
            pickle.dump(user_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(results_u_path, 'rb') as handle:
            user_dict = pickle.load(handle)
    return user_dict


def update_record(results_u, uid, record):
    if uid in results_u:
        if len(results_u[uid]) >= 10:
            results_u[uid].popleft()
        results_u[uid].append(record)
    else:
        results_u[uid] = deque(record)
    return results_u


def feature_engineering(df_):

    #CFG.features = CFG.cate_cols + CFG.cont_cols + [TARGET]
    #
    # df_.loc[:, 'lagged_time'] = (df_[['user_id', 'timestamp']].groupby('user_id')['timestamp'].diff() / 300e3).clip(upper=1)
    # df_.loc[:, 'prior_question_elapsed_time'] = (df_.loc[:, 'prior_question_elapsed_time'] / 300e3).clip(upper=1)
    # import feather
    # df_ = feather.read_dataframe('./input/raw/train.feather')
    # prior_question_elapsed_time_mean = df_.prior_question_elapsed_time.dropna().values.mean()
    # lagged_time_mean = df_.lagged_time.dropna().values.mean()
    # print(prior_question_elapsed_time_mean, lagged_time_mean)
    # del df_
    # gc.collect()
    # prior_question_elapsed_time_mean = 0.091806516
    # lagged_time_mean = 0.22981916761559054
    # print(prior_question_elapsed_time_mean, lagged_time_mean)
    # 0.091806516 0.22981916761559054

    # df_['prior_question_elapsed_time'] = df_.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)
    # df_['lagged_time'] = df_.lagged_time.fillna(lagged_time_mean)

    df_['user_count'] = df_.groupby(['user_id']).cumcount()
    # lagged_y = df_[['user_id', 'answered_correctly']].groupby('user_id')['answered_correctly'].shift(fill_value=0)
    # df_.loc[:, 'rolling_avg_lagged_y'] = lagged_y.cumsum() / (1+df_.loc[:, 'sum_y'])
    return df_


def  add_new_features(df_, settings, CFG, **kwargs):
    questions_df, lectures_df, mappers_dict, results_c = generate_files(settings, CFG, **kwargs)

    sample_indices = get_samples(df_)

    # df_ = df_.set_index('content_id')
    df_ = pd.concat([df_.reset_index(drop=True), questions_df.reindex(df_['content_id'].values).reset_index(drop=True)],
                  axis=1)
    df_ = pd.concat([df_.reset_index(drop=True), lectures_df.reindex(df_['content_id'].values).reset_index(drop=True)],
                  axis=1)
    df_ = pd.concat([df_.reset_index(drop=True), results_c.reindex(df_['content_id'].values).reset_index(drop=True)],
                  axis=1)
    df_['type_of'].fillna(value='None', inplace=True)
    df_['part'].fillna(value=-1, inplace=True)
    df_['answered_correctly_content'].fillna(value=0.5, inplace=True)
    # df_ = feature_engineering(df_)

    return df_, mappers_dict, sample_indices


def preprocess_data(df_, settings, CFG, **kwargs):

    CFG.cate_cols = CFG.cate_cols
    CFG.cont_cols = CFG.cont_cols
    submission_flag = kwargs.get('submission', False)
    if not submission_flag:
        df_.sort_values(['user_id', 'timestamp'], ascending=True, inplace=True)
    else:
        df_.sort_values(['user_id', 'timestamp'], ascending=True, inplace=True)
    df_.reset_index(inplace=True)

    df_, mappers_dict, sample_indices = add_new_features(df_, settings, CFG, **kwargs)
    df_, cate_offset = transform_df(df_, CFG, mappers_dict)

    return df_, sample_indices, cate_offset


def transform_df(df_, CFG, mappers_dict):
    cate_cols = CFG.cate_cols
    cont_cols = CFG.cont_cols

    cate_offset = 1
    for col in cate_cols:
        cate2idx = mappers_dict[col]
        df_.loc[:, col] = df_[col].map(cate2idx).fillna(0).astype(int)
        cate_offset += len(cate2idx)
    for col in cont_cols:
        df_[col].fillna(0, inplace=True)
    return df_[['user_id']+CFG.features], cate_offset


def save_to_feather(file_name="validation-v0-00000000000", output_file_name="validation_v0", max_=30, settings=None):
    fs = gcsfs.GCSFileSystem(token=settings['TOKEN_DIR'])
    output_file_path = os.path.join(settings['CLEAN_DATA_DIR'], f'{output_file_name}.feather')
    if os.path.isfile(output_file_path):
        print('file_name saved!')
        return
    dataset = pd.DataFrame()
    for i in range(max_):
        file_path = f'gs://ktracing/{file_name}{i}.csv'
        if not fs.exists(file_path):
            break
        with fs.open(file_path) as f:
            print(i)
            df = pd.read_csv(f, dtype=dtypes, index_col=False)
            print('current shape:', df.shape)
            dataset = pd.concat([dataset, df])
            print('updated shape:', dataset.shape)
    print('overall shape:', dataset.shape)
    dataset.reset_index(drop=True).to_feather(output_file_path)


# generate sample indices
def get_samples(df_):
    row_id_list = df_[df_.content_type_id==False].index

    sample_indices = []
    df_users = df_.groupby('user_id').groups
    for user_idx, start_indices in df_users.items():
        curr_cnt = 0
        for num, curr_index in enumerate(start_indices):
            # if curr_index in row_id_list:

            sample_indices.append((user_idx, curr_cnt, curr_index))
            curr_cnt += 1

            # else:
            #     assert df_.loc[curr_index, TARGET] == -1

    # df_lens = df_[df_.content_type_id==False].groupby('user_id').size().to_dict()
    return sample_indices


def convert_feather(settings):
    path = settings['RAW_DATA_DIR']
    files = [f for f in os.listdir(path) if f.endswith('.feather')]
    for f in files:
        pd_ = feather.read_dataframe(os.path.join(path, f))
        pd_.to_pickle(os.path.join(path, f.replace('.feather','.pkl')))


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

    for step, (cate_x, cont_x, response, mask, y) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        idx = y >= 0
        cate_x, cont_x, response, mask, y = cate_x[idx], cont_x[idx], response[idx], mask[idx], y[idx]
        cate_x, cont_x, response, mask, y = cate_x.cuda(), cont_x.cuda(), response.cuda(), mask.cuda(), y.cuda()
        batch_size = cate_x.size(0)

        # compute loss
        pred = model(cate_x, cont_x, response, mask)
        loss = torch.nn.BCELoss()(pred, y.reshape(-1,1))

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
        except Exception as e:  # work on python 3.x
            print('Failed: ' + str(e))
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
    print('training batch_time:', batch_time.sum)
    print('training overall losses:', losses.avg)
    print('training overall auc:', accuracies.avg)

    return losses.avg, accuracies.avg



def validate(valid_loader, model):
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluation mode
    model.eval()

    end = time.time()

    predictions = []
    ground_truth = []
    for step, (cate_x, cont_x, response, mask, y) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        idx = y >= 0
        cate_x, cont_x, response, mask, y = cate_x[idx], cont_x[idx], response[idx], mask[idx], y[idx]

        cate_x, cont_x, response, mask, y = cate_x.cuda(), cont_x.cuda(), response.cuda(), mask.cuda(), y.cuda()
        batch_size = cate_x.size(0)

        # compute loss
        with torch.no_grad():
            pred = model(cate_x, cont_x, response, mask)
            auc = metrics.roc_auc_score(y.detach().cpu().numpy(), pred.detach().cpu().numpy())
            losses.update(auc, batch_size)
        predictions.append(pred.detach().cpu())
        ground_truth.append(y.cpu())


    predictions = torch.cat(predictions).numpy()
    ground_truths = torch.cat(ground_truth).numpy()
    print('validation total eval time:', data_time.sum)
    return losses.avg, predictions, ground_truths


def test(valid_loader, model):
    model.eval()
    predictions = []
    for step, (cate_x, cont_x, response, mask, y) in enumerate(valid_loader):

        idx = (y == 0)

        cate_x, cont_x, response, mask = cate_x[idx], cont_x[idx], response[idx], mask[idx]

        cate_x, cont_x, response, mask = cate_x.cuda(), cont_x.cuda(), response.cuda(), mask.cuda()

        with torch.no_grad():
            pred = model(cate_x, cont_x, response, mask)

        predictions.append(pred.detach().cpu())
    predictions = torch.cat(predictions).numpy()
    return predictions


def parse_model_name(CFG, model_name):
    cfg_dict = dict([tok.split('-') for tok in model_name.split('_')])
    CFG.encoder = cfg_dict['a']
    CFG.seq_len = int(cfg_dict['len'])
    CFG.emb_size = int(cfg_dict['e'])
    CFG.hidden_size = int(cfg_dict['h'])
    CFG.nlayers = int(cfg_dict['l'])
    CFG.nheads = int(cfg_dict['hd'])
    return CFG


def load_model(settings, CFG, model_name):

    model_path = os.path.join(settings['MODEL_DIR'], model_name)
    model = encoders[CFG.encoder](CFG)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    # print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))
    model.cuda()
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_lr():
    return scheduler.get_lr()[0]


def get_dataloader(df_, settings, CFG, **kwargs):
    train_df, train_samples, cate_offset = \
        preprocess_data(df_, settings, CFG, submission=kwargs.get('submission', False))
    # print('finish preprocessing')
    # assert cate_offset == 13790


    CFG.total_cate_size = cate_offset
    train_db = KTDataset(CFG, train_df[CFG.features].values, train_samples,
                         CFG.features, user_dict=kwargs.get('user_dict', {}),
                         aug=kwargs.get('aug', CFG.aug),
                         prior_df=kwargs.get('prior_df', None), submission=kwargs.get('submission', False))

    if not kwargs.get('submission', False):
        del train_df
        gc.collect()

    train_loader = DataLoader(
        train_db, batch_size=CFG.batch_size, shuffle=False,
        num_workers=0, pin_memory=True)
    if kwargs.get('submission', False):
        return train_loader, train_df, len(train_samples), train_db.user_dict
    else:
        return train_loader, None, len(train_samples), train_db.user_dict
def update_params(CFG, parameters):
    for key, value in parameters.items():
        setattr(CFG, key, value)
    return CFG


def run_validation(df_, settings, CFG, model_name="", user_dict={}):

    CFG = parse_model_name(CFG, model_name)

    valid_loader, _, _, _ = get_dataloader(df_, settings, CFG, user_dict=user_dict, submission=False)

    model = load_model(settings, CFG, model_name)

    auc_score, prediction, groundtruth = validate(valid_loader, model)
    auc = metrics.roc_auc_score(groundtruth, prediction)
    print('Validation AUC loss: ', auc)
    return prediction, groundtruth


def run_submission(test_batch, settings, CFG, model_name, **kwargs):
    if 'user_dict' in kwargs:
        user_dict = kwargs['user_dict']
    else:
        user_dict = get_user_dict(settings, submission_flag=True)
    test_loader, test_df, _, user_dict = \
        get_dataloader(test_batch, settings, CFG,
                       user_dict=user_dict, prior_df=kwargs.get('prior_df', None), submission=True)
    df_batch_prior = test_df[['user_id'] + CFG.features]
    predictions = run_test(test_loader, settings, CFG, model_name=model_name)

    return predictions, df_batch_prior, user_dict


def run_test(valid_loader, settings=None, CFG=None, model_name=""):

    CFG = parse_model_name(CFG, model_name)

    model = load_model(settings, CFG, model_name)

    prediction = test(valid_loader, model)
    return prediction


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


def adjust_learning_rate(optimizer, epoch, CFG):
    # lr  = CFG.learning_rate
    lr = CFG.lr_decay ** (epoch // 10) * CFG.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr




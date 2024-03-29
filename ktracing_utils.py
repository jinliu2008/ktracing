import os
import yaml

import pandas as pd
import gcsfs
import feather
from sklearn import metrics

from tqdm import tqdm as tqdm_notebook

from collections import deque
import torch

import pickle
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


def generate_files(settings=None, parameters=None):
    questions_df = get_questions_df(settings)
    cate_cols = parameters['cate_cols']
    mappers_dict_path = os.path.join(settings["CLEAN_DATA_DIR"], 'mappers_dict.pkl')
    if not os.path.isfile(mappers_dict_path):
        df_ = feather.read_dataframe(os.path.join(settings["RAW_DATA_DIR"], 'train_v0.feather'))
        questions_df = get_questions_df(settings)
        df_ = df_.join(questions_df, on='left')
        df_.reset_index(inplace=True)
        mappers_dict = {}
        cate_offset = 1
        for col in (cate_cols):
            cate2idx = {}
            for v in df_[col].unique():
                if (v != v) | (v == None): continue
                cate2idx[v] = len(cate2idx) + cate_offset
            mappers_dict[col] = cate2idx
            cate_offset += len(cate2idx)
        with open(mappers_dict_path, 'wb') as handle:
            pickle.dump(mappers_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(mappers_dict_path, 'rb') as handle:
        mappers_dict = pickle.load(handle)

    results_c_path = os.path.join(settings["CLEAN_DATA_DIR"], 'results_c.pkl')
    if not os.path.isfile(results_c_path):
        df_ = feather.read_dataframe(os.path.join(settings["RAW_DATA_DIR"], 'train_v0.feather'))
        results_c = df_.groupby(['content_id']).agg(['mean'])
        results_c.columns = ["answered_correctly_content"]
        with open(results_c_path, 'wb') as handle:
            pickle.dump(results_c, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(results_c_path, 'rb') as handle:
        results_c = pickle.load(handle)

    # results_u_path = os.path.join(settings["CLEAN_DATA_DIR"], 'results_u.pkl')

    return questions_df, mappers_dict, results_c


def get_user_dict(settings, user_list=[]):
    df_ = feather.read_dataframe(os.path.join(settings["RAW_DATA_DIR"], 'train_v0.feather'))
    # df_ = df_[df_.answered_correctly != -1]
    if user_list:
        df_ = df_[df_.user_id.isin(user_list)].groupby('user_id').tail(CFG.window_size)
    else:
        df_ = df_.groupby('user_id').tail(CFG.window_size)
    # df_.reset_index(drop=True).to_feather(results_u_path)
    results_u = {uid: u for uid, u in df_.groupby('user_id')}
    return results_u


def update_record(results_u, uid, record):
    if uid in results_u:
        if len(results_u[uid]) >= 10:
            results_u[uid].popleft()
        results_u[uid].append(record)
    else:
        results_u[uid] = deque(record)
    return results_u


def feature_engineering(df_):
    df_.loc[:, 'lagged_time'] = df_[['user_id', 'timestamp']].groupby('user_id')['timestamp'].diff() / 1e3
    df_.loc[:, 'prior_question_elapsed_time'] = df_.loc[:, 'prior_question_elapsed_time'] / 1e3
    lagged_y = df_[['user_id', 'answered_correctly']].groupby('user_id')['answered_correctly'].shift(fill_value=0)
    df_.loc[:, 'sum_y'] = df_.groupby('user_id')['user_id'].cumcount()
    df_.loc[:, 'rolling_avg_lagged_y'] = lagged_y.cumsum() / (1+df_.loc[:, 'sum_y'])
    return df_


def add_features(df_, settings, parameters, mode='train'):
    questions_df, mappers_dict, results_c = generate_files(settings=settings, parameters=parameters)
    df_.sort_values(['user_id','timestamp'], ascending=True, inplace=True)
    if mode == 'validation':
        results_u = get_user_dict(settings, user_list=df_.user_id.unique().tolist())
        selected_users = {user: results_u[user] for user in results_u if user in df_.user_id}
        df_ = pd.concat(list(selected_users.values())+[df_], axis=0)
        sample_indices = get_sample_indices(df_, results_u)
    else:
        sample_indices = get_sample_indices(df_)

    # df_ = df_.set_index('content_id')
    df_ = pd.concat([df_.reset_index(drop=True), questions_df.reindex(df_['content_id'].values).reset_index(drop=True)],
                  axis=1)
    df_ = pd.concat([df_.reset_index(drop=True), results_c.reindex(df_['content_id'].values).reset_index(drop=True)],
                  axis=1)

    df_ = feature_engineering(df_)
    return df_, mappers_dict, sample_indices


def add_features_validate(df_, settings, parameters):
    questions_df, mappers_dict, results_c, results_u = generate_files(settings=settings, parameters=parameters)
    df_.sort_values(['timestamp'], ascending=True, inplace=True)

    selected_users = {user: results_u[user] for user in df_.users}
    df_ = pd.concat(list(selected_users.values()), df_, axis=0)

    df_ = df_.set_index('content_id')
    df_ = df_.join(questions_df, how='left')
    df_ = df_.join(results_c, how='left')
    df_.reset_index(inplace=True)

    # df_users = df_.groupby('user_id').groups
    # df_[['sum_y', 'rolling_avg_lagged_y']] = 0
    # for user_idx, start_indices in df_users.items():
    #     df_np = np.zeros((len(start_indices), 2))
    #     for i, idx in enumerate(start_indices):
    #         if i == 0:
    #             if user_idx in results_u:
    #                 prev_sum = results_u[user_idx]["sum_y"] + 1
    #                 prev_avg = results_u[user_idx]["rolling_avg_lagged_y"]
    #             else:
    #                 prev_sum = 1
    #                 prev_avg = 0
    #         else:
    #             prev_avg = \
    #                 (prev_avg * prev_sum + df_.loc[prev_idx, 'answered_correctly'])/(prev_sum+1)
    #             prev_sum = prev_sum + 1
    #
    #         prev_idx = idx
    #         df_np[i, :] = [prev_sum, prev_avg]
    #     df_[df_['user_id'] == user_idx][['sum_y', 'rolling_avg_lagged_y']] = df_np

    sample_indices = get_sample_indices(df_, results_u)
    return df_, mappers_dict, sample_indices


def preprocess(settings=None, parameters=None, mode='train', update_flag=False, output_file=""):
    output_file_path = os.path.join(settings["CLEAN_DATA_DIR"], output_file)
    if os.path.isfile(output_file_path) and not update_flag:
        return

    if mode == 'train':
        file_name = settings['TRAIN_DATASET']
        df_ = feather.read_dataframe(os.path.join(settings['RAW_DATA_DIR'], file_name))
        # logic to add features

    if mode == 'validation':
        file_name = settings['VALIDATION_DATASET']
        df_ = feather.read_dataframe(os.path.join(settings['RAW_DATA_DIR'], file_name))
    df_.sort_values(['timestamp'], ascending=True, inplace=True)
    df_, mappers_dict, sample_indices = add_features(df_, settings, parameters, mode=mode)
    df_, cate_offset = transform_df(df_, parameters, mappers_dict)
    if output_file:
        torch.save([df_, sample_indices, cate_offset], output_file_path)

    return df_, sample_indices, cate_offset


def transform_df(df_, parameters, mappers_dict):
    cate_cols = parameters['cate_cols']
    cont_cols = parameters['cont_cols']

    cate_offset = 1
    for col in cate_cols:
        cate2idx = mappers_dict[col]
        df_.loc[:, col] = df_[col].map(cate2idx).fillna(0).astype(int)
        cate_offset += len(cate2idx)
    for col in cont_cols:
        df_[col].fillna(0, inplace=True)
    return df_, cate_offset


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


# generate train sample indices
def get_sample_indices(df_, results_u=None):
    # df_.set_index('row_id', inplace=True)
    # df_.set_index('row_id', inplace=True)
    row_id_list = df_[df_.content_type_id==False].index
    sample_indices = []
    df_users = df_.groupby('user_id').groups
    for user_idx, start_indices in df_users.items():
        if isinstance(results_u, dict) and user_idx in results_u:
            start_idx = len(results_u[user_idx])
        else:
            start_idx = 0
        for num, curr_index in enumerate(start_indices):
            if (curr_index in row_id_list) and (num >= start_idx):
                sample_indices.append((user_idx, num))
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
    for step, (cate_x, cont_x, mask, y) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        cate_x, cont_x, mask, y = cate_x.cuda(), cont_x.cuda(), mask.cuda(), y.cuda()
        batch_size = cate_x.size(0)

        # compute loss
        with torch.no_grad():
            pred = model(cate_x, cont_x, mask)
            try:
                auc = metrics.roc_auc_score(y.detach().cpu().numpy(), pred.detach().cpu().numpy())
                losses.update(auc, batch_size)
            except Exception as e:
                print('Failed: ' + str(e))
        predictions.append(pred.detach().cpu())
        ground_truth.append(y.cpu())

    predictions = torch.cat(predictions).numpy()
    ground_truths = torch.cat(ground_truth).numpy()
    print('validation total eval time:', data_time.sum)
    return losses.avg, predictions, ground_truths


def test(valid_loader, model):
    model.eval()
    predictions = []
    for step, (cate_x, cont_x, mask, _) in enumerate(valid_loader):
        cate_x, cont_x, mask = cate_x.cuda(), cont_x.cuda(), mask.cuda()

        with torch.no_grad():
            pred = model(cate_x, cont_x, mask)
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
    print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))
    model.cuda()
    return model


def run_validation(settings=None, parameters=None, CFG=None, model_name="", df_=None):

    CFG = parse_model_name(CFG, model_name)
    (valid_df, valid_samples, cate_offset) = \
        preprocess(settings=settings, parameters=parameters, mode='validation', update_flag=True)

    CFG.total_cate_size = cate_offset
    CFG.cate_cols = parameters['cate_cols']
    CFG.cont_cols = parameters['cont_cols']

    valid_db = KTDataset(CFG, valid_df, valid_samples)
    valid_loader = DataLoader(
        valid_db, batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=True)

    model = load_model(settings, CFG, model_name)

    auc_score, prediction, groundtruth = validate(valid_loader, model)
    auc = metrics.roc_auc_score(groundtruth, prediction)
    print('Validation AUC loss: ', auc)


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




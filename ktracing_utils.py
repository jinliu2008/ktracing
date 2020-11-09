import os
import yaml

import pandas as pd
import gcsfs
import feather

import os
import json
from tqdm import tqdm as tqdm_notebook
import torch

import os
import json
import pickle
import logging

import time


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
    learning_rate = 3.0e-3
    batch_size = 1024
    num_workers = 4
    print_freq = 100
    test_freq = 1
    start_epoch = 0
    num_train_epochs = 3
    warmup_steps = 1
    max_grad_norm = 100
    gradient_accumulation_steps = 1
    weight_decay = 0.01
    dropout = 0.2
    emb_size = 20
    hidden_size = 20
    nlayers = 2
    nheads = 10
    seq_len = 20
    target_size = 1
    res_dir = 'res_dir_0'


def get_logger(settings):

    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(filename=os.path.join(settings['LOGS_DIR'], f'log_{timestr}.txt'),
                        format=FORMAT, level=logging.INFO)
    logger = logging.getLogger('main')
    # logger.setLevel(logging.info)


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
    file_name = f'b-{batch_size}_a-{CFG.encoder}_e-{CFG.emb_size}_h-{CFG.hidden_size}_'
    f'd-{CFG.dropout}_l-{CFG.nlayers}_hd-{CFG.nheads}_'
    f's-{CFG.seed}_len-{CFG.seq_len}_aug-{CFG.aug}_da-{CFG.input_filename}'
    return file_name


def generate_files(settings={}, parameters={}):
    cate_cols = parameters['cate_cols']
    mappers_dict_path = os.path.join(settings["CLEAN_DATA_DIR"], 'mappers_dict.pkl')
    if not os.path.isfile(mappers_dict_path):
        df_ = feather.read_dataframe(os.path.join(settings["RAW_DATA_DIR"], 'train_v0.feather'))
        questions_df = pd.read_csv(os.path.join(settings['RAW_DATA_DIR'], 'questions.csv'), usecols=[0, 3],
                                   dtype={'question_id': 'int16', 'part': 'int8'})
        questions_df.rename(columns={'question_id': 'content_id'}, inplace=True)
        df_ = pd.merge(df_, questions_df, on='content_id', how='left')
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
        results_c = df_[['content_id', 'answered_correctly']].groupby(['content_id']).agg(['mean'])
        results_c.columns = ["answered_correctly_content"]
        with open(results_c_path, 'wb') as handle:
            pickle.dump(results_c, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(results_c_path, 'rb') as handle:
        results_c = pickle.load(handle)

    return mappers_dict, results_c


def preprocess(settings={}, parameters={}, mode='train', update_flag=False, output_file=None):
    if mode == 'train':
        file_name = settings['TRAIN_DATASET']
    elif mode == 'validation':
        file_name = settings['VALIDATION_DATASET']
    elif mode == "test":
        file_name = settings['TEST_DATASET']
    else:
        raise NotImplementedError

    if not output_file:
        output_file = file_name.split('.')[0] + '.pt'
    output_file_path = os.path.join(settings["CLEAN_DATA_DIR"], output_file)
    if os.path.isfile(output_file_path) and not update_flag:
        return

    df_ = feather.read_dataframe(os.path.join(settings['RAW_DATA_DIR'], file_name))
    df_.sort_values(['user_id', 'timestamp'], inplace=True)

    questions_df = pd.read_csv(os.path.join(settings['RAW_DATA_DIR'], 'questions.csv'), usecols=[0, 3],
                                dtype={'question_id': 'int16', 'part': 'int8'})
    questions_df.rename(columns={'question_id': 'content_id'}, inplace=True)
    df_ = pd.merge(df_, questions_df, on='content_id', how='left')

    mappers_dict, results_c = generate_files(settings=settings, parameters=parameters)
    cate_cols = parameters['cate_cols']
    cont_cols = parameters['cont_cols']
    seq_len = parameters['seq_len']
    sample_indices = get_sample_indices(df_, seq_len=seq_len)
    cate_offset = 1
    for col in cate_cols:
        cate2idx = mappers_dict[col]
        df_.loc[:, col] = df_[col].map(cate2idx).fillna(0).astype(int)
        cate_offset += len(cate2idx)
    df_ = pd.merge(df_, results_c, on=['content_id'], how="left")

    df_.loc[:, 'lagged_time'] = df_[['user_id', 'timestamp']].groupby('user_id')['timestamp'].diff() / 1e3
    df_.loc[:, 'prior_question_elapsed_time'] = df_.loc[:, 'prior_question_elapsed_time'] / 1e6

    for col in cont_cols:
        df_[col].fillna(0, inplace=True)

    torch.save([df_, sample_indices, cate_offset], output_file_path)

    return df_, sample_indices, cate_offset


def save_to_feather(file_name="validation-v0-00000000000", output_file_name="validation_v0", max_=30, settings={}):
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
def get_sample_indices(df_, seq_len=50):
    df = df_[df_.content_type_id == False]
    df.set_index('row_id', inplace=True)
    sample_indices = []
    df_users = df.groupby('user_id').groups
    for user_idx, start_indices in enumerate(tqdm_notebook(df_users.values())):
        for num, curr_index in enumerate(start_indices):
            sample_indices.append((user_idx, num))
    return sample_indices


def convert_feather(settings):
    path =settings['RAW_DATA_DIR']
    files = [f for f in os.listdir(path) if f.endswith('.feather')]
    for f in files:
        pd_ = feather.read_dataframe(os.path.join(path, f))
        pd_.to_pickle(os.path.join(path, f.replace('.feather','.pkl')))
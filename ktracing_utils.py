import os
import yaml

import pandas as pd
import gcsfs
import feather
from sklearn import metrics

from tqdm import tqdm as tqdm_notebook
import torch

import pickle
import logging
import math

import time
import ktracing_data
import ktracing_models
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
    learning_rate = 3.0e-3
    batch_size = 1024
    num_workers = 4
    print_freq = 100
    test_freq = 1
    start_epoch = 0
    num_train_epochs = 1
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
    seed = 123
    data_seed = 123
    encoder = 'TRANSFORMER'
    aug = 0.0


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
        results_c = df_[['content_id', 'answered_correctly']].groupby(['content_id']).agg(['mean'])
        results_c.columns = ["answered_correctly_content"]
        with open(results_c_path, 'wb') as handle:
            pickle.dump(results_c, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(results_c_path, 'rb') as handle:
        results_c = pickle.load(handle)
    return mappers_dict, results_c


def preprocess(questions_df=None, mappers_dict=None, results_c=None,
               settings=None, parameters=None, mode='train', update_flag=False, output_file=None):
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
    df_.sort_values(['timestamp'], ascending=True, inplace=True)
    if mode == 'train' or mode == 'validation':
        questions_df = get_questions_df(settings)
        mappers_dict, results_c = generate_files(settings=settings, parameters=parameters)
    df_ = df_.join(questions_df, how='left')
    df_ = df_.join(results_c, how='left')
    df_.reset_index(inplace=True)

    cate_cols = parameters['cate_cols']
    cont_cols = parameters['cont_cols']
    sample_indices = get_sample_indices(df_)
    cate_offset = 1
    for col in cate_cols:
        cate2idx = mappers_dict[col]
        df_.loc[:, col] = df_[col].map(cate2idx).fillna(0).astype(int)
        cate_offset += len(cate2idx)

    df_.loc[:, 'lagged_time'] = df_[['user_id', 'timestamp']].groupby('user_id')['timestamp'].diff() / 1e3
    df_.loc[:, 'prior_question_elapsed_time'] = df_.loc[:, 'prior_question_elapsed_time'] / 1e6

    for col in cont_cols:
        df_[col].fillna(0, inplace=True)

    if mode == 'train' or mode == 'validation':
        torch.save([df_, sample_indices, cate_offset], output_file_path)

    return df_, sample_indices, cate_offset


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
def get_sample_indices(df_):
    df = df_[df_.content_type_id == False]
    df.set_index('row_id', inplace=True)
    sample_indices = []
    df_users = df.groupby('user_id').groups
    for user_idx, start_indices in enumerate(tqdm_notebook(df_users.values())):
        for num, curr_index in enumerate(start_indices):
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

        auc = metrics.roc_auc_score(y.detach().cpu().numpy(), pred.detach().cpu().numpy())
        accuracies.update(auc, batch_size)

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

            auc = metrics.roc_auc_score(y.detach().cpu().numpy(), pred.detach().cpu().numpy())
            losses.update(auc, batch_size)
            if step%20==0:
                print('running time:', data_time.avg)
                print('auc: ', auc, 'avg: ', losses.avg)

        predictions.append(pred.detach().cpu())
        ground_truth.append(y.cpu())

    predictions = torch.cat(predictions).numpy()
    ground_truths = torch.cat(ground_truth).numpy()

    return losses.avg, predictions, ground_truths


def run_validation(settings=None, parameters=None, CFG=None, model_name=""):
    (valid_df, valid_samples, cate_offset) = \
        preprocess(settings=settings, parameters=parameters, mode='validation', update_flag=True)
    cfg_dict = dict([tok.split('-') for tok in model_name.replace('bowl_', '').split('_')])
    CFG.encoder = cfg_dict['a']
    CFG.seq_len = int(cfg_dict['len'])
    CFG.emb_size = int(cfg_dict['e'])
    CFG.hidden_size = int(cfg_dict['h'])
    CFG.nlayers = int(cfg_dict['l'])
    CFG.nheads = int(cfg_dict['hd'])
    CFG.total_cate_size = cate_offset
    CFG.cate_cols = parameters['cate_cols']
    CFG.cont_cols = parameters['cont_cols']

    model_path = os.path.join(settings['MODEL_DIR'], model_name)
    model = ktracing_models.encoders[CFG.encoder](CFG)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))
    model.cuda()
    print('cfg:', CFG.__dict__)
    valid_db = ktracing_data.KTDataset(CFG, valid_df, valid_samples)
    valid_loader = DataLoader(
        valid_db, batch_size=CFG.batch_size, shuffle=False,
        num_workers=0, pin_memory=True)
    auc_score, prediction, groundtruth = validate(valid_loader, model)
    auc = metrics.roc_auc_score(groundtruth, prediction)
    print('current loss: ', auc)


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
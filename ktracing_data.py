# import warnings
# warnings.filterwarnings('ignore')

from collections import defaultdict
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import gc
TARGET = 'answered_correctly'

# funcs for user stats with loop
def add_user_feats(df, answered_correctly_sum_u_dict, count_u_dict):
    acsu = np.zeros(len(df), dtype=np.int32)
    cu = np.zeros(len(df), dtype=np.int32)
    for cnt,row in enumerate(tqdm(df[['user_id','answered_correctly']].values)):
        acsu[cnt] = answered_correctly_sum_u_dict[row[0]]
        cu[cnt] = count_u_dict[row[0]]
        answered_correctly_sum_u_dict[row[0]] += row[1]
        count_u_dict[row[0]] += 1
    user_feats_df = pd.DataFrame({'answered_correctly_sum_u':acsu, 'count_u':cu})
    user_feats_df['answered_correctly_avg_u'] = user_feats_df['answered_correctly_sum_u'] / user_feats_df['count_u']
    df = pd.concat([df, user_feats_df], axis=1)
    return df

def add_user_feats_without_update(df, answered_correctly_sum_u_dict, count_u_dict):
    acsu = np.zeros(len(df), dtype=np.int32)
    cu = np.zeros(len(df), dtype=np.int32)
    for cnt,row in enumerate(df[['user_id']].values):
        acsu[cnt] = answered_correctly_sum_u_dict[row[0]]
        cu[cnt] = count_u_dict[row[0]]
    user_feats_df = pd.DataFrame({'answered_correctly_sum_u':acsu, 'count_u':cu})
    user_feats_df['answered_correctly_avg_u'] = user_feats_df['answered_correctly_sum_u'] / user_feats_df['count_u']
    df = pd.concat([df, user_feats_df], axis=1)
    return df

def update_user_feats(df, answered_correctly_sum_u_dict, count_u_dict):
    for row in df[['user_id','answered_correctly','content_type_id']].values:
        if row[2] == 0:
            answered_correctly_sum_u_dict[row[0]] += row[1]
            count_u_dict[row[0]] += 1

def fill_zeros_with_last(arr):
    prev = np.arange(len(arr))
    prev[arr == 0] = 0
    prev = np.maximum.accumulate(prev)
    return arr[prev]

class KTDataset(Dataset):
    def __init__(self, cfg, df, sample_indices, columns, user_dict={}, aug=0.0, aug_p=0.5
                 , prior_df=None, submission=False):
        self.cfg = cfg
        self.df = df

        self.sample_indices = sample_indices
        self.seq_len = self.cfg.seq_len
        self.aug = aug
        self.aug_p = aug_p
        self.cate_cols = self.cfg.cate_cols
        self.cont_cols = self.cfg.cont_cols

        self.user_dict = user_dict
        self.start_token = 2
        self.columns = columns
        self.prior_df = prior_df
        self.submission = submission
        if isinstance(prior_df, pd.DataFrame):
            assert self.submission == True
            # update dict
            for user_id, u in prior_df.groupby('user_id'):
                curr_row = u[self.columns].values
                if user_id not in self.user_dict:
                    curr_array = curr_row.copy()
                else:
                    user_hist = self.user_dict[user_id]

                    if "user_count" in self.columns:
                        user_count_idx = self.columns.index('user_count')
                        last_value = user_hist[-1, user_count_idx]+1
                        curr_row[:,user_count_idx] += last_value

                    curr_array = np.concatenate((user_hist, curr_row), axis=0)

                # update dict
                if curr_array.shape[0] > self.seq_len:
                    self.user_dict[user_id] = curr_array[-self.seq_len:, :]
                else:
                    self.user_dict[user_id] = curr_array.copy()

    def __getitem__(self, idx):
        # 275030867
        user_id, user_idx, index = self.sample_indices[idx]

        if self.aug > 0:
            if len_ > 50:
                if np.random.binomial(1, self.aug_p) == 1:
                    cut_ratio = np.random.rand()
                    if cut_ratio > self.aug:
                        cut_ratio = self.aug
                    len_ = max(int(len_ * cut_ratio), 30)

        curr_row = np.array(self.df[index,:]).reshape(1,-1)

        if user_id not in self.user_dict:
            curr_array = curr_row.copy()
        else:
            user_hist = self.user_dict[user_id]

            if "user_count" in self.columns:
                user_count_idx = self.columns.index('user_count')
                curr_row[0, user_count_idx] = user_hist[-1, user_count_idx] + 1

            curr_array = np.concatenate((user_hist, curr_row), axis=0)

        # update dict
        if not self.submission:
            if curr_array.shape[0] > self.seq_len:
                self.user_dict[user_id] = curr_array[-self.seq_len:,:].copy()
            else:
                self.user_dict[user_id] = curr_array.copy()

        target_idx = self.columns.index(TARGET)
        # curr_array[:, target_idx] = np.roll(curr_array[:, target_idx], 1)

        len_ = min(self.seq_len, curr_array.shape[0])
        curr_array =curr_array[-len_:,:]

        curr_array[-1, target_idx] = self.start_token

        cate_df = curr_array[:, :len(self.cate_cols)]

        #prior elaspse time
        if 'prior_question_elapsed_time' in self.columns:
            elpase_time_idx = self.columns.index('prior_question_elapsed_time')
            curr_array[:, elpase_time_idx] = np.clip(curr_array[:, elpase_time_idx]/300e3, 0, 1)
            # curr_array[0, elpase_time_idx] = 0.091806516

        boolean_idx = []

        if 'prior_question_had_explanation' in self.columns:
            had_explanation_idx = self.columns.index('prior_question_had_explanation')
            boolean_idx.append(had_explanation_idx)

        if 'timestamp' in self.columns:
            timestamp_idx = self.columns.index('timestamp')
            curr_array[1:, timestamp_idx] = \
                fill_zeros_with_last(np.clip(np.diff(curr_array[:, timestamp_idx])/300e3, 0, 1))
            curr_array[0, timestamp_idx] = 0.22981916761559054

        cols = len(self.columns)
        cont_df = curr_array[:, len(self.cate_cols):-1].copy()

        if "user_count" in self.columns:
            user_count_idx = self.columns.index('user_count')-len(self.cate_cols)
            cont_df[:, user_count_idx] = np.clip(cont_df[:, user_count_idx]/1e3, 0, 1)

        for c in range(len(self.cate_cols), cols-1):
            if c in boolean_idx:
                continue
            cont_df[:, c-len(self.cate_cols)] = np.log1p(cont_df[:, c-len(self.cate_cols)].astype(float))

        response_df = curr_array[:, -1:]
        target_df = curr_row[-1, -1]

        # prepare cate
        tmp_cate_x = torch.LongTensor(cate_df.astype(float))
        cate_x = torch.LongTensor(self.seq_len, len(self.cate_cols)).zero_()

        cate_x[-len_:,:] = tmp_cate_x[-len_:,:]

        tmp_cont_x = torch.FloatTensor(cont_df.astype(float))
        cont_x = torch.FloatTensor(self.seq_len, len(self.cont_cols)).zero_()
        cont_x[-len_:,:] = tmp_cont_x[-len_:,:]

        tmp_response = torch.LongTensor(response_df.astype(float))
        response = torch.LongTensor(self.seq_len, 1).zero_()
        response[-len_:,:] = tmp_response[-len_:,:]
        response[response < 0] = 3

        mask = torch.ByteTensor(self.seq_len).zero_()
        mask[-len_:] = 1

        if target_df is not None:
            target = torch.FloatTensor(np.array(target_df).astype(float))
        else:
            target = 0

        return cate_x, cont_x, response, mask, target

    def __len__(self):
        return len(self.sample_indices)

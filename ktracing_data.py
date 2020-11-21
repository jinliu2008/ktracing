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


class KTDataset(Dataset):
    def __init__(self, cfg, df, sample_indices, user_dict, columns, aug=0.0, aug_p=0.5, prior_df=None):
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
        self.submission = False
        if isinstance(prior_df, pd.DataFrame):
            self.submission = True
            # update dict
            for user_id, u in prior_df.groupby('user_id'):
                curr_row = u[self.columns]
                if user_id not in self.user_dict:
                    curr_array = curr_row.copy()
                else:
                    user_hist = self.user_dict[user_id]
                    curr_array = np.concatenate((user_hist, curr_row), axis=0)

                # update dict
                if curr_array.shape[0] > self.seq_len:
                    self.user_dict[user_id] = curr_array[-self.seq_len:, :]
                else:
                    self.user_dict[user_id] = curr_array.copy()

    def __getitem__(self, idx):

        user_id, user_idx, index = self.sample_indices[idx]

        if self.aug > 0:
            if len_ > 50:
                if np.random.binomial(1, self.aug_p) == 1:
                    cut_ratio = np.random.rand()
                    if cut_ratio > self.aug:
                        cut_ratio = self.aug
                    len_ = max(int(len_ * cut_ratio), 30)

        curr_row = np.array(self.df[index,:]).reshape(1,-1)
        target_idx = self.columns.index(TARGET)
        if user_id not in self.user_dict:
            curr_array = curr_row.copy()
        else:
            user_hist = self.user_dict[user_id]
            curr_array = np.concatenate((user_hist, curr_row), axis=0)

        # update dict
        if not self.submission:
            if curr_array.shape[0] > self.seq_len:
                self.user_dict[user_id] = curr_array[-self.seq_len:,:]
            else:
                self.user_dict[user_id] = curr_array.copy()

        curr_array[:, target_idx] = np.roll(curr_array[:, target_idx], 1)

        len_ = min(self.seq_len, curr_array.shape[0])
        curr_array =curr_array[-len_:,:]
        curr_array[0, target_idx] = self.start_token

        cate_df = curr_array[:, :len(self.cate_cols)]
        cont_df = curr_array[:, len(self.cate_cols):]
        target_df = curr_row[-1, -1]

        # prepare cate
        tmp_cate_x = torch.LongTensor(cate_df.astype(float))
        cate_x = torch.LongTensor(self.seq_len, len(self.cate_cols)).zero_()

        cate_x[-len_:,:] = tmp_cate_x[-len_:,:]

        tmp_cont_x = torch.FloatTensor(cont_df.astype(float))
        cont_x = torch.FloatTensor(self.seq_len, len(self.cont_cols)+1).zero_()
        cont_x[-len_:,:] = tmp_cont_x[-len_:,:]

        mask = torch.ByteTensor(self.seq_len).zero_()
        mask[-len_:] = 1

        if target_df is not None:
            target = torch.FloatTensor(np.array(target_df))
        else:
            target = 0

        return cate_x, cont_x, mask, target

    def __len__(self):
        return len(self.sample_indices)

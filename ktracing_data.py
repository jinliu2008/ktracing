# import warnings
# warnings.filterwarnings('ignore')


import torch
from torch.utils.data import Dataset
import numpy as np
import gc
TARGET = ['answered_correctly']


class KTDataset(Dataset):
    def __init__(self, cfg, df, sample_indices, aug=0.0, aug_p=0.5):
        self.cfg = cfg
        # self.df = df.copy()
        # if "row_id" in self.df:
        #     self.df = self.df.set_index('row_id')
        self.sample_indices = sample_indices
        self.seq_len = self.cfg.seq_len
        self.aug = aug
        self.aug_p = aug_p
        self.cate_cols = self.cfg.cate_cols
        self.cont_cols = self.cfg.cont_cols
        self.df_users, self.cate_df, self.cont_df, self.target_df = {}, {}, {}, {}
        df_users = df.groupby('user_id').groups
        # self.df_users_len = df_users.copy()
        # self.df_users_np = df_users.copy()
        for user_idx, start_indices in df_users.items():
            curr_user = df.loc[start_indices]
            self.cate_df[user_idx] = curr_user[self.cate_cols].values
            self.cont_df[user_idx] = np.log1p(curr_user[self.cont_cols].values.clip(min=0))
            self.target_df[user_idx] = curr_user[TARGET].values

        del df, df_users
        gc.collect()
        # self.cate_df = df[self.cate_cols].values
        # self.cont_df = self.df[self.cont_cols].values
        # self.target_df = self.df[TARGET].values
        #self.cont_df = np.log1p(self.df[self.cont_cols])


    def __getitem__(self, idx):

        user_id, index = self.sample_indices[idx]
        # curr_len = len(self.df_user_np[user_id])
        # if curr_len == 0:
        #     if index == 0:
        #         default_value = 2
        #     # default value
        #         self.df_user_np[user_id].append(default_value)
        # else:
        #     if curr_len>=10:
        #         self.df_user_np[user_id].popleft()
        #     new_value = (self.df_user_np[user_id][-1]*(curr_len-1)+self.df.loc[index-1, TARGET].values)/curr_len
        #     self.df_user_np[user_id].append(new_value)

        len_ = min(self.seq_len, index+1)
        # indices = self.df_users[user_id][index+1-len_:index+1]
        if self.aug > 0:
            if len_ > 50:
                if np.random.binomial(1, self.aug_p) == 1:
                    cut_ratio = np.random.rand()
                    if cut_ratio > self.aug:
                        cut_ratio = self.aug
                    len_ = max(int(len_ * cut_ratio), 30)

        # len_ = min(seq_len, len(indices))

        tmp_cate_x = torch.LongTensor(self.cate_df[user_id][index+1-len_:index+1, :])
        cate_x = torch.LongTensor(self.seq_len, len(self.cate_cols)).zero_()
        cate_x[-len_:] = tmp_cate_x[-len_:]

        tmp_cont_x = torch.FloatTensor(self.cont_df[user_id][index+1-len_:index+1, :])
        cont_x = torch.FloatTensor(self.seq_len, len(self.cont_cols)).zero_()
        cont_x[-len_:] = tmp_cont_x[-len_:]

        mask = torch.ByteTensor(self.seq_len).zero_()
        mask[-len_:] = 1

        if self.target_df is not None:
            target = torch.FloatTensor(self.target_df[user_id][index])
        else:
            target = 0
        #print(cate_x.shape, cont_x.shape, target.shape)
        return cate_x, cont_x, mask, target

    def __len__(self):
        return len(self.sample_indices)

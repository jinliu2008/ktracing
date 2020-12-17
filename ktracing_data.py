# import warnings
# warnings.filterwarnings('ignore')

from collections import defaultdict
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import gc
TARGET = 'answered_correctly'
MAX_SEQ = 100


class SAKTDataset(Dataset):
    def __init__(self, group, n_skill, max_seq=MAX_SEQ):  # HDKIM 100
        super(SAKTDataset, self).__init__()
        self.max_seq = max_seq
        self.n_skill = n_skill
        self.samples = group

        #         self.user_ids = [x for x in group.index]
        self.user_ids = []
        for user_id in group.index:
            q, qa = group[user_id]
            if len(q) < 5:  # HDKIM 10
                continue
            self.user_ids.append(user_id)

            # HDKIM Memory reduction
            if len(q) > self.max_seq:
                group[user_id] = (q[-self.max_seq:], qa[-self.max_seq:])

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, qa_ = self.samples[user_id]
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        if seq_len >= self.max_seq:
            q[:] = q_[-self.max_seq:]
            qa[:] = qa_[-self.max_seq:]
        else:
            q[-seq_len:] = q_
            qa[-seq_len:] = qa_

        target_id = q[1:]
        label = qa[1:]

        x = np.zeros(self.max_seq - 1, dtype=int)
        x = q[:-1].copy()
        x += (qa[:-1] == 1) * self.n_skill

        return x, target_id, label


class TestDataset(Dataset):
    def __init__(self, samples, test_df, skills, max_seq=MAX_SEQ):  # HDKIM 100
        super(TestDataset, self).__init__()
        self.samples = samples
        self.user_ids = [x for x in test_df["user_id"].unique()]
        self.test_df = test_df
        self.skills = skills
        self.n_skill = len(skills)
        self.max_seq = max_seq

    def __len__(self):
        return self.test_df.shape[0]

    def __getitem__(self, index):
        test_info = self.test_df.iloc[index]

        user_id = test_info["user_id"]
        target_id = test_info["content_id"]

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)

        if user_id in self.samples.index:
            q_, qa_ = self.samples[user_id]

            seq_len = len(q_)

            if seq_len >= self.max_seq:
                q = q_[-self.max_seq:]
                qa = qa_[-self.max_seq:]
            else:
                q[-seq_len:] = q_
                qa[-seq_len:] = qa_

        x = np.zeros(self.max_seq - 1, dtype=int)
        x = q[1:].copy()
        x += (qa[1:] == 1) * self.n_skill

        questions = np.append(q[2:], [target_id])

        return x, questions
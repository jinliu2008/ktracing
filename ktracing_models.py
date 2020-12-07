# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
from transformers.modeling_albert import AlbertConfig, AlbertModel
from transformers.modeling_xlnet import XLNetConfig, XLNetModel
from transformers.modeling_xlm import XLMConfig, XLMModel
from transformers.modeling_gpt2 import GPT2Model, GPT2Config, GPT2PreTrainedModel, Block
import math
from torch.autograd import Variable

from torch.autograd import Variable


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransfomerModel(nn.Module):
    def __init__(self, cfg):
        super(TransfomerModel, self).__init__()
        self.cfg = cfg
        cate_col_size = len(cfg.cate_cols)
        cont_col_size = len(cfg.cont_cols)+1

        self.cate_emb = nn.Embedding(cfg.total_cate_size, cfg.emb_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(cfg.total_cate_size, cfg.hidden_size)

        # self.position_emb_nn = nn.Embedding(cfg.emb_size, cfg.hidden_size)
        self.cate_proj = nn.Sequential(
            nn.Linear(cfg.emb_size * cate_col_size, cfg.hidden_size // 2),
            nn.LayerNorm(cfg.hidden_size // 2),
        )

        self.cont_emb = nn.Sequential(
            nn.Linear(cont_col_size, cfg.hidden_size // 2),
            nn.LayerNorm(cfg.hidden_size // 2),
        )

        self.position_embeddings = nn.Embedding(cfg.total_cate_size, cfg.hidden_size)

        self.config = BertConfig(
            # 3,  # not used
            hidden_size=cfg.hidden_size,
            num_hidden_layers=cfg.nlayers,
            num_attention_heads=cfg.nheads,
            intermediate_size=cfg.hidden_size,
            hidden_dropout_prob=cfg.dropout,
            attention_probs_dropout_prob=cfg.dropout,
        )
        self.encoder = BertEncoder(self.config)

        def get_reg():
            return nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.Dropout(cfg.dropout),
                nn.ReLU(),
                # nn.Linear(cfg.hidden_size, cfg.hidden_size),
                # nn.LayerNorm(cfg.hidden_size),
                # nn.Dropout(cfg.dropout),
                # nn.ReLU(),
                nn.Linear(cfg.hidden_size, cfg.target_size),
            )

        self.reg_layer = get_reg()

    def forward(self, cate_x, cont_x, mask):
        batch_size = cate_x.size(0)

        # cate_x = self.cate_pos_encoder(cate_x)
        cate_emb = self.cate_emb(cate_x).view(batch_size, self.cfg.seq_len, -1)
        cate_emb = self.cate_proj(cate_emb)

        # cont_x = self.cont_pos_encoder(cont_x)
        cont_emb = self.cont_emb(cont_x)
        seq_emb = torch.cat([cate_emb, cont_emb], 2)

        seq_length = self.cfg.seq_len
        position_ids = torch.arange(seq_length, dtype=torch.long, device=cate_x.device)
        position_ids = position_ids.unsqueeze(0).expand((batch_size, seq_length))
        position_emb = self.position_embeddings(position_ids)
        seq_emb = (seq_emb + position_emb)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.config.num_hidden_layers

        encoded_layers = self.encoder(seq_emb, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]
        sequence_output = sequence_output[:, -1]

        pred_y = self.reg_layer(sequence_output)
        return torch.sigmoid(pred_y)

class LSTMModel(nn.Module):
    def __init__(self, cfg):
        super(LSTMModel, self).__init__()
        self.cfg = cfg
        cate_col_size = len(cfg.cate_cols)
        cont_col_size = len(cfg.cont_cols)
        self.cate_emb = nn.Embedding(cfg.total_cate_size, cfg.emb_size, padding_idx=0)
        self.cate_proj = nn.Sequential(
            nn.Linear(cfg.emb_size *cate_col_size, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2)
        )
        self.cont_emb = nn.Sequential(
            nn.Linear(cont_col_size, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
        )

        self.encoder = nn.LSTM(cfg.hidden_size,
                               cfg.hidden_size, cfg.nlayers, dropout=cfg.dropout, batch_first=True)

        def get_reg():
            return nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.Dropout(cfg.dropout),
                nn.ReLU(),
                nn.Linear(cfg.hidden_size, cfg.target_size),
            )
        self.reg_layer = get_reg()

    def forward(self, cate_x, cont_x, mask):
        batch_size = cate_x.size(0)

        cate_emb = self.cate_emb(cate_x).view(batch_size, self.cfg.seq_len, -1)
        cate_emb = self.cate_proj(cate_emb)
        cont_emb = self.cont_emb(cont_x)

        seq_emb = torch.cat([cate_emb, cont_emb], 2)

        _, (h, c) = self.encoder(seq_emb)
        sequence_output = h[-1]

        pred_y = self.reg_layer(sequence_output)
        return torch.sigmoid(pred_y)


encoders = {
    'LSTM': LSTMModel,
    'TRANSFORMER': TransfomerModel,
    # 'BERT': DSB_BertModel,
    # 'GPT2': DSB_GPT2Model,
    # 'ALBERT': DSB_ALBERTModel,
    # 'XLNET': DSB_XLNetModel,
    # 'XLM': DSB_XLMModel,
}
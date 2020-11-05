import torch
import torch.nn as nn


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
    # 'TRANSFORMER': TransfomerModel,
    # 'BERT': DSB_BertModel,
    # 'GPT2': DSB_GPT2Model,
    # 'ALBERT': DSB_ALBERTModel,
    # 'XLNET': DSB_XLNetModel,
    # 'XLM': DSB_XLMModel,
}

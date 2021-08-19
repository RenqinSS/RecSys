import numpy as np
import torch
import torch.nn as nn


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, n_user, n_item, cfg):
        super(SASRec, self).__init__()

        self.n_user = n_user
        self.n_item = n_item
        self.dev = cfg.device

        # TODO: loss += cfg.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = nn.Embedding(self.n_item+1, cfg.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.maxlen, cfg.hidden_units) # TO IMPROVE
        self.emb_dropout = nn.Dropout(p=cfg.dropout_rate)

        self.attn_layernorms = nn.ModuleList() # to be Q for self-attention
        self.attn_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        for _ in range(cfg.n_blocks):
            self.attn_layernorms.append(nn.LayerNorm(cfg.hidden_units, eps=1e-8))
            self.attn_layers.append(nn.MultiheadAttention(cfg.hidden_units, cfg.n_heads, cfg.dropout_rate, batch_first=True))
            self.forward_layernorms.append(nn.LayerNorm(cfg.hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(cfg.hidden_units, cfg.dropout_rate))
        
        self.last_layernorm = nn.LayerNorm(cfg.hidden_units, eps=1e-8)


    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        pad_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~pad_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attn_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attn_layers)):
            Q = self.attn_layernorms[i](seqs)
            attn_output, attn_output_weights = self.attn_layers[i](Q, seqs, seqs, attn_mask=attn_mask)
            seqs = Q + attn_output

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~pad_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        final_feat = self.log2feats(log_seqs)[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits
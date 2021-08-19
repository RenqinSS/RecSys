from yacs.config import CfgNode
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from data.datasets.movielen_1m import data_partition, WarpSampler
from models.SASRec import SASRec
from evaluate.eval import evaluate, evaluate_valid



if __name__ == '__main__':
    with open(r'configs/movielen_1m/SASRec.yaml') as f:
        cfg = CfgNode.load_cfg(f)

    dataset = data_partition(cfg.dataset)
    user_trn, user_val, user_tes, n_user, n_item = dataset
    n_batch = len(user_trn) // cfg.batch_size

    avg_trn_len = sum([len(user_trn[u]) for u in user_trn]) / len(user_trn)
    print('average sequence length: %.2f' % avg_trn_len)

    sampler = WarpSampler(user_trn, n_user, n_item, batch_size=cfg.batch_size, maxlen=cfg.maxlen, n_workers=3)
    model = SASRec(n_user, n_item, cfg).to(cfg.device) # no ReLU activation in original SASRec implementation?

    for name, param in model.named_parameters():
        if 'weight' in name and 'layernorm' not in name:
            torch.nn.init.xavier_normal_(param.data)

    model.train()
    epoch_start_idx = 1
    if cfg.weights:
        model.load_state_dict(torch.load(cfg.weights, map_location=torch.device(cfg.device)))
        #tail = cfg.weights[cfg.weights.find('epoch=') + 6:]
        #epoch_start_idx = int(tail[:tail.find('.')]) + 1

    if cfg.infer_only:
        model.eval()
        t_test = evaluate(model, dataset, cfg)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()

    for epoch in range(epoch_start_idx, cfg.n_epochs + 1):
        if cfg.infer_only: break
        for step in range(n_batch):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=cfg.device), torch.zeros(neg_logits.shape, device=cfg.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            q = pos_logits[indices]
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += cfg.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs


        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, cfg)
            t_valid = evaluate_valid(model, dataset, cfg)
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            # f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            # f.flush()
            t0 = time.time()
            model.train()

        # if epoch == cfg.n_epochs:
            # folder = cfg.dataset + '_' + cfg.train_dir
            # fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            # fname = fname.format(cfg.n_epochs, cfg.lr, cfg.num_blocks, cfg.num_heads, cfg.hidden_units, cfg.maxlen)
            # torch.save(model.state_dict(), os.path.join('ckpts', ))

    # f.close()
    sampler.close()
    print("Done")

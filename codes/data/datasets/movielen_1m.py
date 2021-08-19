import sys
import copy
import torch
import random
import os
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_trn = {}
    user_val = {}
    user_tes = {}
    # assume user/item index starting from 1
    with open('data/%s.txt' % fname, 'r') as f:
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_trn[user] = User[user]
            user_val[user] = []
            user_tes[user] = []
        else:
            user_trn[user] = User[user][:-2]
            user_val[user] = [User[user][-2]]
            user_tes[user] = [User[user][-1]]
    return [user_trn, user_val, user_tes, usernum, itemnum]


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_trn, usernum, itemnum, batch_size, maxlen, queue, SEED):
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_trn[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_trn[user][-1]
        idx = maxlen - 1

        ts = set(user_trn[user])
        for i in reversed(user_trn[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = [sample() for i in range(batch_size)]
        queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(Process(target=sample_function, args=(
                User, usernum, itemnum, batch_size, maxlen, self.queue, np.random.randint(2e9))))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
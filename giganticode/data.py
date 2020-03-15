import collections
import os
import pickle
from typing import List

import torch


class Vocab():
    "Contain the correspondence between numbers and tokens and numericalize."
    def __init__(self, itos):
        self.itos = itos
        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})

    def numericalize(self, t) -> List[int]:
        "Convert a list of tokens `t` to their ids."
        return [self.stoi[w] for w in t]

    def textify(self, nums, sep=' ') -> List[str]:
        "Convert a list of `nums` to their tokens."
        return sep.join([self.itos[i] for i in nums]) if sep is not None else [self.itos[i] for i in nums]

    def __getstate__(self):
        return {'itos':self.itos}

    def __setstate__(self, state:dict):
        self.itos = state['itos']
        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})

    def save(self, path):
        "Save `self.itos` in `path`"
        pickle.dump(self.itos, open(path, 'wb'))

    @classmethod
    def load(cls, path):
        "Load the `Vocab` contained in `path`"
        itos = pickle.load(open(path, 'rb'))
        return cls(itos)


class Dictionary(object):
    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    @property
    def word2idx(self):
        return self.vocab.stoi

    @property
    def idx2word(self):
        return self.vocab.itos

    def __len__(self):
        return len(self.vocab.itos)


class Corpus(object):
    def __init__(self, path: str):
        self.path = path
        self.dictionary = Dictionary(Vocab.load(os.path.join(path, 'vocab.pkl')))

    def _get_numericalized_tensor(self, file: str):
        with open(os.path.join(self.path, 'numericalized', file), 'r') as f:
            return torch.LongTensor(list(map(lambda x: int(x), f.read().split(" "))))

    @property
    def train(self):
        return self._get_numericalized_tensor('train.txt')

    @property
    def test(self):
        return self._get_numericalized_tensor('test.txt')

    @property
    def valid(self):
        return self._get_numericalized_tensor('valid.txt')
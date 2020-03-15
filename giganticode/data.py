import os

import torch
from fastai.text import Vocab


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
            return torch.tensor(list(map(lambda x: int(x), f.read().split(" "))))

    def train(self):
        return self._get_numericalized_tensor('train.txt')

    def test(self):
        return self._get_numericalized_tensor('test.txt')

    def valid(self):
        return self._get_numericalized_tensor('valid.txt')
import collections
import os
import pickle
from typing import List, Union, Optional

import torch
from langmodels.cuda_util import get_device
from langmodels.file_util import get_all_files
from tqdm import tqdm

from codeprep.api.corpus import nosplit, PreprocessedCorpus


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


def numericalize_corpus(path: str, vocab: Vocab,
                        device: Union[int, str] = get_device(),
                        output_file: Optional[str] = None) -> torch.Tensor:

    all_files = [f for f in get_all_files(path, None)]

    all_numbers = []
    for file in tqdm(all_files):
        with file.open('r') as f:
            all_numbers.extend(vocab.numericalize(f.read().split(" ")))

    if output_file:
        with open(output_file, 'w') as f:
            f.write(" ".join(map(lambda x: str(x), all_numbers)))
    return torch.tensor(all_numbers, device=device, dtype=torch.int64)


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
    def _prep_corpus(self, path: str) -> PreprocessedCorpus:
        base_path, dir = os.path.split(path)
        output_path = os.path.join(base_path, dir + '_prepped')
        corpus = nosplit(path, no_com=True, no_str=True, no_unicode=True, no_spaces=True, calc_vocab=False, output_path=output_path)
        return corpus

    def __init__(self, path: str):
        self.path = path
        corpus = self._prep_corpus(path)
        vocab = Vocab(corpus.load_vocab().keys())
        self.dictionary = Dictionary(vocab)
        self.train = numericalize_corpus(os.path.join(corpus.path_to_prep_dataset, 'train'), vocab)
        self.valid = numericalize_corpus(os.path.join(corpus.path_to_prep_dataset, 'valid'), vocab)
        self.test = numericalize_corpus(os.path.join(corpus.path_to_prep_dataset, 'test'), vocab)
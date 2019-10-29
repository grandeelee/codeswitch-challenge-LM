from logging import getLogger
import math
import numpy as np
import torch
from my_dictionary import BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD

logger = getLogger(__name__)


class Dataset(object):
    """
    mainly takes care of batched indexed data preparation
    prepare batch according to the longest sequence, shape is (n , slen), slen is max(len) + 2
    batch size fixed if tokens_per_batch ==-1
    or prepare batch to contain an approx number of tokens
    remove_long_sentences(max_len) can be called before get_iterator
    """

    def __init__(self, sent, pos, dictionary, params):
        self.batch_size = params.batch_size
        self.max_batch_size = params.batch_size
        self.dictionary = dictionary
        self.sent = sent
        self.pos = pos
        # the length of each sent, without counting the eos index
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        # remove empty sentneces
        self.remove_empty_sentences()
        # sanity checks
        self.check()

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.pos)

    def check(self):
        """
        Sanity checks.
        """
        eos = self.dictionary.phoneme2id[EOS_WORD]
        assert len(self.pos) == np.sum(self.sent[self.pos[:, 1]] == eos)  # check sentences indices
        assert self.lengths.min() > 0  # check empty sentences

    def batch_sentences(self, sentences, direction='forward'):
        """
        Take as input a list of n sentences (torch.LongTensor vectors) and return
        a tensor of size (n, slen) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        [[BOS w1 w2 ... wn EOS PAD]
         [BOS w1 ... wn EOS PAD PAD]]
        """
        pad_idx = self.dictionary.phoneme2id[PAD_WORD]
        eos_idx = self.dictionary.phoneme2id[EOS_WORD]
        bos_idx = self.dictionary.phoneme2id[BOS_WORD]

        lengths = torch.LongTensor([len(s) + 2 for s in sentences])
        sent = torch.LongTensor(lengths.size(0), lengths.max().item()).fill_(pad_idx)
        if direction == 'forward':
            # set the start of the sequence to be eos and the end to be a pad
            sent[:, 0] = bos_idx
            for i, s in enumerate(sentences):
                sent[i, 1:lengths[i] - 1].copy_(torch.from_numpy(s.astype(np.int64)))
                sent[i, lengths[i] - 1] = eos_idx
        if direction == 'backward':
            sent[:, 0] = eos_idx
            for i, s in enumerate(sentences):
                sent[i, 1:lengths[i] - 1].copy_(torch.from_numpy(s[::-1].astype(np.int64)))
                sent[i, lengths[i] - 1] = bos_idx

        return sent, lengths

    def remove_unk_sentences(self, unk_tol=0.0):
        """
        Remove sentences with more than 20% unknown tokens
        """
        unk_idx = self.dictionary.phoneme2id[UNK_WORD]
        init_size = len(self.pos)
        indices = []
        for idx in range(init_size):
            if np.sum(self.sent[self.pos[idx, 0]: self.pos[idx, 1]] == unk_idx) == 0:
                indices.append(idx)
        self.pos = self.pos[indices]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        logger.info("Removed %i sentences with unk token." % (init_size - len(indices)))
        self.check()

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.pos)
        indices = np.arange(len(self.pos))
        indices = indices[self.lengths[indices] > 0]
        self.pos = self.pos[indices]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))
        self.check()

    def remove_long_sentences(self, max_len):
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len >= 0
        if max_len == 0:
            return
        init_size = len(self.pos)
        indices = np.arange(len(self.pos))
        indices = indices[self.lengths[indices] <= max_len]
        self.pos = self.pos[indices]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        logger.info("Removed %i too long sentences." % (init_size - len(indices)))
        self.check()

    def select_data(self, a, b):
        """
        Only select a subset of the dataset.
        select a subset of positions
        and DELETE THE REST, USE WITH CAUTION
        """
        assert 0 <= a < b <= len(self.pos)
        logger.info("Selecting sentences from %i to %i ..." % (a, b))

        # sub-select
        self.pos = self.pos[a:b]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]

        # re-index
        min_pos = self.pos.min()
        max_pos = self.pos.max()
        self.pos -= min_pos
        self.sent = self.sent[min_pos:max_pos + 1]

        # sanity checks
        self.check()

    def get_batches_iterator(self, batches, direction='forward'):
        """
        Return a sentences iterator, given the associated sentence batches.
        """
        assert direction in ['forward', 'backward'], 'direction can only be forward or backward'

        for sentence_ids in batches:
            # if 0 < self.max_batch_size < len(sentence_ids):
            #     np.random.shuffle(sentence_ids)
            #     sentence_ids = sentence_ids[:self.max_batch_size]
            pos = self.pos[sentence_ids]
            sent = [self.sent[a:b] for a, b in pos]
            sent = self.batch_sentences(sent, direction)
            yield sent

    def get_iterator(self, shuffle, group_by_size=False, return_indices=False, direction='forward'):
        """
        Return a sentences iterator.
        """
        assert direction in ['forward', 'backward'], 'direction can only be forward or backward'
        n_sentences = len(self.pos)
        assert type(shuffle) is bool and type(group_by_size) is bool

        # sentence lengths
        lengths = self.lengths + 2

        # select sentences to iterate over
        if shuffle:
            indices = np.random.permutation(len(self.pos))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            # TODO uncheck this during training
            # indices = indices[np.argsort(lengths[indices], kind='mergesort')]
            indices = indices[np.argsort(lengths[indices], kind='mergesort')[::-1]]

        # create batches - either have a fixed number of sentences, or a similar number of tokens
        batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))

        # optionally shuffle batches
        if shuffle:
            np.random.shuffle(batches)

        # sanity checks
        assert n_sentences == sum([len(x) for x in batches])
        assert lengths[indices].sum() == sum([lengths[x].sum() for x in batches])

        # return the iterator
        return self.get_batches_iterator(batches, direction)


if __name__ == '__main__':

    file = '../../data/en.test.pth'
    data = torch.load(file)
    data = Dataset(data['sentences'], data['positions'], data['dictionary'], None)
    gen = data.get_iterator(True)
    for batch in gen:
        continue

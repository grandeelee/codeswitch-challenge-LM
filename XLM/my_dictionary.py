import os
from collections import Counter, OrderedDict
import numpy as np
from tqdm import tqdm
import mmap
import torch
from logging import getLogger

logger = getLogger(__name__)


def get_num_lines(file):
    """
    a helper function to get the total number of lines from file read quickly
    :param file:
    :return:
    """
    fp = open(file, 'r+')
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


# consider using these special characters for all subsequent dict
BOS_WORD = '<s>'
EOS_WORD = '</s>'
PAD_WORD = '<pad>'
UNK_WORD = '<unk>'

SPECIAL_WORD = '<special%i>'
N_SPECIAL_WORDS = 10

SEP_WORD = SPECIAL_WORD % 0
MASK_WORD = SPECIAL_WORD % 1


class Dictionary(object):
    """
    1) before using anything from this class make sure to tokenize the data corpus according
    to the desired format, it return indexed data with the dict.
    2) Dictionary can be initialized from count file, self.from_vocab
        or get dictionary from corpus -> 1) self.from_corpus
    3) self.id2word, word2id, counts are all dictionaries
    4) perform self.check_valid() if unsure
    5) implement index_data which could be overwritten if calls for other format
    6)  data['dictionary'] contains id2word word2id and counts
        data['positions'] contains shape: (n_sent, 2) where each item is (start, stop) of sent
        data['sentences'] contains shape: (n_words,) sents are separated by eos index
        data['unk_words'] contains the stats of unknown word distribution, (n_words, counts)
    """

    def __init__(self):
        self.id2word = {}
        self.word2id = {}
        self.counts = Counter()

    def __len__(self):
        """
        Returns the number of words in the dictionary.
        """
        return len(self.id2word)

    def __getitem__(self, i):
        """
        Returns the word of the specified index.
        """
        return self.id2word[i]

    def __contains__(self, w):
        """
        Returns whether a word is in the dictionary.
        """
        return w in self.word2id

    def __eq__(self, y):
        """
        Compare this dictionary with another one.
        """
        self.check_valid()
        y.check_valid()
        if len(self.id2word) != len(y):
            return False
        return all(self.id2word[i] == y[i] for i in range(len(y)))

    def from_corpus(self, path, min_occur=0, most_freq=-1):
        """
        path accept file path or a list of file paths
        :param path:
        :param min_occur: default 0
        :param most_freq: default -1
        :return:
        """
        # if one big corpus
        if isinstance(path, str):
            assert os.path.isfile(path), path
            data = open(path, 'r', encoding='utf-8')
            for line in tqdm(data, total=get_num_lines(path)):
                # the file has to be preprocessed with the desired tokenizer
                self.counts.update(line.split())
            data.close()
        # if split into train test valid
        if isinstance(path, list):
            for p in path:
                assert os.path.isfile(p), p
                data = open(p, 'r', encoding='utf-8')
                for line in tqdm(data, total=get_num_lines(p)):
                    # the file has to be preprocessed with the desired tokenizer
                    self.counts.update(line.split())
                data.close()
        # sort counts into descending order
        ordered_counts = OrderedDict(sorted(self.counts.items(), key=lambda item: (-item[1], item[0])))
        # takes care of special tokens
        self.word2id = {BOS_WORD: 0, EOS_WORD: 1, PAD_WORD: 2, UNK_WORD: 3}
        for i in range(N_SPECIAL_WORDS):
            self.word2id[SPECIAL_WORD % i] = 4 + i
        self.counts = {k: 0 for k in self.word2id.keys()}
        for k, v in ordered_counts.items():
            if len(self.id2word) == most_freq:
                break
            if ordered_counts[k] > min_occur:
                if k != UNK_WORD:
                    self.word2id[k] = len(self.word2id)
                self.counts[k] = v
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.check_valid()

    def from_vocab(self, vocab_path):
        """
        Create a dictionary from a vocabulary file. the vocab file must be in word count format
        """
        skipped = 0
        assert os.path.isfile(vocab_path), vocab_path
        # takes care of special tokens, if the count file has no special token
        self.word2id = {BOS_WORD: 0, EOS_WORD: 1, PAD_WORD: 2, UNK_WORD: 3}
        for i in range(N_SPECIAL_WORDS):
            self.word2id[SPECIAL_WORD % i] = 4 + i
        self.counts = {k: 0 for k in self.word2id.keys()}
        # read in vocab file
        f = open(vocab_path, 'r', encoding='utf-8')
        for i, line in enumerate(f):
            if '\u2028' in line:
                skipped += 1
                continue
            line = line.rstrip().split()
            if len(line) != 2:
                skipped += 1
                continue
            assert len(line) == 2, (i, line)
            assert line[1].isdigit(), (i, line)
            if line[0] in self.word2id:
                skipped += 1
                logger.debug('%s already in vocab' % line[0])
                continue
            if not line[1].isdigit():
                skipped += 1
                logger.debug('Empty word at line %s with count %s' % (i, line))
                continue
            self.word2id[line[0]] = 4 + N_SPECIAL_WORDS + i - skipped  # shift because of extra words
            self.counts[line[0]] = int(line[1])
        f.close()
        self.id2word = {v: k for k, v in self.word2id.items()}
        logger.info("Read %i words from the vocabulary file." % len(self.id2word))
        if skipped > 0:
            logger.warning("Skipped %i empty lines!" % skipped)
        self.check_valid()

    def check_valid(self):
        """
        Check that the dictionary is valid.
        """
        # check the special tokens
        assert self.word2id[BOS_WORD] == 0
        assert self.word2id[EOS_WORD] == 1
        assert self.word2id[PAD_WORD] == 2
        assert self.word2id[UNK_WORD] == 3
        assert all(self.id2word[4 + i] == SPECIAL_WORD % i for i in range(N_SPECIAL_WORDS))
        # check the words
        assert len(self.id2word) == len(self.word2id) == len(self.counts)
        assert set(self.word2id.keys()) == set(self.counts.keys())
        # check the word list and index tally
        for i in range(len(self.id2word)):
            assert self.word2id[self.id2word[i]] == i
        # check it is sorted
        last_count = 1e18
        for i in range(4 + N_SPECIAL_WORDS, len(self.id2word) - 1):
            count = self.counts[self.id2word[i]]
            assert count <= last_count
            last_count = count

    def index(self, word, no_unk=False):
        """
        Returns the index of the specified word.
        """
        if no_unk:
            return self.word2id[word]
        else:
            return self.word2id.get(word, self.word2id[UNK_WORD])

    def max_vocab(self, max_vocab):
        """
        Limit the vocabulary size.
        """
        assert max_vocab >= 1
        init_size = len(self)
        self.id2word = {k: v for k, v in self.id2word.items() if k < max_vocab}
        self.word2id = {v: k for k, v in self.id2word.items()}
        self.counts = {k: v for k, v in self.counts.items() if k in self.word2id}
        self.check_valid()
        logger.info("Maximum vocabulary size: %i. Dictionary size: %i -> %i (removed %i words)."
                    % (max_vocab, init_size, len(self), init_size - len(self)))

    def min_count(self, min_count):
        """
        Threshold on the word frequency counts.
        """
        assert min_count >= 0
        init_size = len(self)
        self.id2word = {k: v for k, v in self.id2word.items() if
                        self.counts[self.id2word[k]] >= min_count or k < 4 + N_SPECIAL_WORDS}
        self.word2id = {v: k for k, v in self.id2word.items()}
        self.counts = {k: v for k, v in self.counts.items() if k in self.word2id}
        self.check_valid()
        logger.info("Minimum frequency count: %i. Dictionary size: %i -> %i (removed %i words)."
                    % (min_count, init_size, len(self), init_size - len(self)))

    def write_vocab(self, path):
        """
        write a list to file separated by '\n' in the format of word count
        """
        with open(path, mode='w', encoding='utf-8') as f:
            f.writelines('{} {}'.format(k, v) + '\n' for k, v in self.counts.items())

    @staticmethod
    def index_para(dico, path_1, path_2, bin_path):
        positions = []
        sentences = []
        unk_words = {}
        eos_index = dico.word2id[EOS_WORD]
        unk_index = dico.word2id[UNK_WORD]
        bos_index = dico.word2id[BOS_WORD]

        f_1 = open(path_1, 'r', encoding='utf-8')
        f_2 = open(path_2, 'r', encoding='utf-8')
        for i, (line_1, line_2) in tqdm(enumerate(zip(f_1, f_2)), total=get_num_lines(path_1)):
            s_1 = line_1.rstrip().split()
            # skip empty sentences
            if len(s_1) == 0:
                logger.debug("Empty sentence in line %i." % i)
                continue
            s_2 = line_2.rstrip().split()
            # skip empty sentences
            if len(s_2) == 0:
                logger.debug("Empty sentence in line %i." % i)
                continue

            count_unk = 0
            indexed = []
            for w in s_1:
                word_id = dico.index(w, no_unk=False)
                # if we find a special word which is not an unknown word, skip the sentence
                if 0 <= word_id < 4 + N_SPECIAL_WORDS and word_id != unk_index:
                    logger.warning('Found unexpected special word "%s" (%i)!!' % (w, word_id))
                    continue
                assert word_id >= 0
                indexed.append(word_id)
                # useful to see the unk distribution
                if word_id == unk_index:
                    unk_words[w] = unk_words.get(w, 0) + 1
                    count_unk += 1

            indexed.extend([eos_index, bos_index])

            for w in s_2:
                word_id = dico.index(w, no_unk=False)
                # if we find a special word which is not an unknown word, skip the sentence
                if 0 <= word_id < 4 + N_SPECIAL_WORDS and word_id != unk_index:
                    logger.warning('Found unexpected special word "%s" (%i)!!' % (w, word_id))
                    continue
                assert word_id >= 0
                indexed.append(word_id)
                # useful to see the unk distribution
                if word_id == unk_index:
                    unk_words[w] = unk_words.get(w, 0) + 1
                    count_unk += 1
            # add sentence
            # a list of start stop
            positions.append([len(sentences), len(sentences) + len(indexed)])
            #  a huge list of index
            sentences.extend(indexed)
            sentences.append(eos_index)  # EOS index
        f_1.close()
        f_2.close()

        # tensorize data
        positions = np.int64(positions)
        if len(dico) < 1 << 16:
            sentences = np.uint16(sentences)
        elif len(dico) < 1 << 31:
            sentences = np.int32(sentences)
        else:
            raise Exception("Dictionary is too big.")
        assert sentences.min() >= 0
        data = {
            'dictionary': dico,
            'positions': positions,
            'sentences': sentences,
            'unk_words': unk_words,
        }
        logger.info("Saving the data to %s ..." % bin_path)
        torch.save(data, bin_path, pickle_protocol=4)

        return data

    @staticmethod
    def index_data(dico, path, bin_path=None):
        """
        Index sentences with a dictionary. The dictionary will not keep the data.
        return one big list of indexed ids, special token are not added at this stage
        so the corpus should only have <unk> and </s> as a special token

        """
        positions = []
        sentences = []
        unk_words = {}
        eos_index = dico.word2id[EOS_WORD]
        unk_index = dico.word2id[UNK_WORD]
        # index sentences
        f = open(path, 'r', encoding='utf-8')
        for i, line in tqdm(enumerate(f), total=get_num_lines(path)):
            s = line.rstrip().split()
            # skip empty sentences
            if len(s) == 0:
                logger.debug("Empty sentence in line %i." % i)
                continue
            # index sentence words
            count_unk = 0
            indexed = []
            for w in s:
                word_id = dico.index(w, no_unk=False)
                # if we find a special word which is not an unknown word, skip the sentence
                if 0 <= word_id < 4 + N_SPECIAL_WORDS and word_id != unk_index:
                    logger.warning('Found unexpected special word "%s" (%i)!!' % (w, word_id))
                    continue
                assert word_id >= 0
                indexed.append(word_id)
                # useful to see the unk distribution
                if word_id == unk_index:
                    unk_words[w] = unk_words.get(w, 0) + 1
                    count_unk += 1
            # add sentence
            # a list of start stop
            positions.append([len(sentences), len(sentences) + len(indexed)])
            #  a huge list of index
            sentences.extend(indexed)
            sentences.append(eos_index)  # EOS index
        f.close()

        # tensorize data
        positions = np.int64(positions)
        if len(dico) < 1 << 16:
            sentences = np.uint16(sentences)
        elif len(dico) < 1 << 31:
            sentences = np.int32(sentences)
        else:
            raise Exception("Dictionary is too big.")
        assert sentences.min() >= 0
        data = {
            'dictionary': dico,
            'positions': positions,
            'sentences': sentences,
            'unk_words': unk_words,
        }
        if bin_path:
            logger.info("Saving the data to %s ..." % bin_path)
            torch.save(data, bin_path, pickle_protocol=4)

        return data

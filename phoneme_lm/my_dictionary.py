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
        self.id2phoneme = {}
        self.phoneme2id = {}
        self.lexicon = {}
        self.counts = Counter()

    def __len__(self):
        """
        Returns the number of words in the dictionary.
        """
        return len(self.id2phoneme)

    def __getitem__(self, i):
        """
        Returns the word of the specified index.
        """
        return self.phoneme2id[i]

    def __contains__(self, w):
        """
        Returns whether a word is in the dictionary.
        """
        return w in self.phoneme2id

    def from_vocab(self, vocab_path):
        """
        Create a dictionary from a vocabulary file. the vocab file must be in word count format
        """
        skipped = 0
        assert os.path.isfile(vocab_path), vocab_path
        # takes care of special tokens, if the count file has no special token
        self.phoneme2id = {BOS_WORD: 0, EOS_WORD: 1, PAD_WORD: 2, UNK_WORD: 3}
        # read in vocab file
        f = open(vocab_path, 'r', encoding='utf-8')
        for i, line in enumerate(f):
            if '\u2028' in line:
                skipped += 1
                continue
            self.counts.update(line.split()[1:])
        f.close()
        ordered_counts = OrderedDict(sorted(self.counts.items(), key=lambda item: (item[0], item[1])))
        for k in ordered_counts.keys():
            self.phoneme2id[k] = len(self.phoneme2id)
        self.id2phoneme = {v: k for k, v in self.phoneme2id.items()}
        logger.info("Read %i words from the vocabulary file." % len(self.id2phoneme))
        if skipped > 0:
            logger.warning("Skipped %i empty lines!" % skipped)

        f = open(vocab_path, 'r', encoding='utf-8')
        for i, line in enumerate(f):
            line = line.rstrip().split()
            self.lexicon[line[0]] = line[1:]
        f.close()

    def write_vocab(self, path):
        """
        write a list to file separated by '\n' in the format of word count
        """
        with open(path + '_phoneme2id', mode='w', encoding='utf-8') as f:
            f.writelines('{} {}'.format(k, v) + '\n' for k, v in enumerate(self.phoneme2id.keys()))
        with open(path + '_word2id', mode='w', encoding='utf-8') as f:
            f.writelines('{} {}'.format(k, v) + '\n' for k, v in enumerate(self.lexicon))
        with open(path + '_word2phoneme', mode='w', encoding='utf-8') as f:
            f.writelines('{} {}'.format(k, v) + '\n' for k, v in enumerate(self.lexicon.values()))


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
        eos_index = dico.phoneme2id[EOS_WORD]
        unk_index = dico.phoneme2id[UNK_WORD]
        # index sentences
        f = open(path, 'r', encoding='utf-8')
        for i, line in tqdm(enumerate(f), total=get_num_lines(path)):
            s = line.rstrip().split()
            # skip empty sentences
            if len(s) == 0:
                logger.debug("Empty sentence in line %i." % i)
            count_unk = 0
            phoneme_list = []
            for w in s:
                phones = dico.lexicon.get(w, [UNK_WORD])
                phoneme_list += phones
                if phones == [UNK_WORD]:
                    unk_words[w] = unk_words.get(w, 0) + 1
                    count_unk += 1
            indexed = [dico.phoneme2id[w] for w in phoneme_list]

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


if __name__ == '__main__':
    dico = Dictionary()
    dico.from_vocab('/home/grandee/projects/LM/data/phoneme_lm/lexicon_mode.txt')
    dico.write_vocab('test.vocab')
    torch.save(dico, '/home/grandee/projects/LM/data/dico.pth', pickle_protocol=4)
    data = dico.index_data(dico, '/home/grandee/projects/LM/phoneme_lm/test.txt', 'test.pth')
    print('done')
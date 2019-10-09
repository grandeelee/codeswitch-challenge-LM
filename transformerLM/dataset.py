import os
from collections import Counter, OrderedDict
from tqdm import tqdm
import mmap
from torch.utils import data
import numpy as np


def get_num_lines(file_path):
	fp = open(file_path, 'r+')
	buf = mmap.mmap(fp.fileno(), 0)
	lines = 0
	while buf.readline():
		lines += 1
	return lines


class Dictionary(object):
	def __init__(self):
		self.word2idx = {}
		self.idx2word = []
		self.counter = Counter()
		self.total = 0

	def write_vocab(self, vocab, path):
		"""
		write a list to file separated by '\n'
		"""
		with open(path, mode='w', encoding='utf-8', errors='surrogateescape') as f:
			f.writelines(i + '\n' for i in vocab)

	def get_count(self, path):
		assert os.path.exists(path)
		data = open(path, encoding='utf-8', errors='surrogateescape')
		for line in tqdm(data, total=get_num_lines(path)):
			self.counter.update(line.split())

	def get_dict(self, min_occur=0, most_freq=-1):
		self.counter = OrderedDict(sorted(self.counter.items(), key=lambda item: (-item[1], item[0])))
		for word in self.counter:
			if len(self.idx2word) == most_freq:
				break
			if self.counter[word] > min_occur:
				self.word2idx[word] = len(self.word2idx)
				self.idx2word.append(word)

		self.total = sum(self.counter.values())

	def __len__(self):
		return len(self.idx2word)


class Corpus(object):
	def __init__(self, path):
		self.dictionary = Dictionary()
		self.dict_init(path)
		self.train = self.tokenize(os.path.join(path, 'train.txt'))
		self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
		self.test = self.tokenize(os.path.join(path, 'test.txt'))

	def dict_init(self, path):
		self.dictionary.get_count(os.path.join(path, 'train.txt'))
		self.dictionary.get_count(os.path.join(path, 'valid.txt'))
		self.dictionary.get_count(os.path.join(path, 'test.txt'))
		self.dictionary.get_dict()

	def tokenize(self, path):
		"""Tokenizes a text file into 2D list according to sentence boundary"""
		ids = []
		with open(path, 'r', encoding='utf-8', errors='surrogateescape') as f:
			for line in tqdm(f, total=get_num_lines(path)):
				id = [self.dictionary.word2idx[word] for word in line.split()]
				ids.append(id)
		return ids


class Dataset(data.Dataset):
	def __init__(self, data, n_ctx, n_vocab):
		"""
		data is a 2D list
		:param data:
		"""
		self.data = data
		self.n_ctx = n_ctx
		self.n_vocab = n_vocab

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		x = np.zeros((self.n_ctx, 2), dtype=np.int32)
		m = np.zeros(self.n_ctx, dtype=np.int32)
		lx = len(self.data[index])
		x[:lx, 0] = self.data[index]
		x[:, 1] = np.arange(self.n_vocab, self.n_vocab + self.n_ctx)
		m[:lx] = 1

		return x, m

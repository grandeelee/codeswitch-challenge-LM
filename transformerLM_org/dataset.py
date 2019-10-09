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
		# add special tokens here
		for token in ['<pad>']:
			self.word2idx[token] = len(self.word2idx)
			self.idx2word.append(token)
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
		self.train = self.input_stream(self.tokenize(os.path.join(path, 'train.txt')), 512)
		self.valid = self.input_stream(self.tokenize(os.path.join(path, 'valid.txt')), 512)
		self.test = self.input_stream(self.tokenize(os.path.join(path, 'test.txt')), 512)

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

	def input_stream(self, ids, n_ctx):
		inputs = []
		# bos = self.dictionary.word2idx['<bos>']
		# eos = self.dictionary.word2idx['<eos>']
		pad = self.dictionary.word2idx['<pad>']
		slen = 0
		sent = []
		for id in ids:
			if slen + len(id) + 1 <= n_ctx:
				sent.extend(id + [pad])
				slen += len(id) + 1
			else:
				inputs.append(sent + [pad] * (n_ctx - len(sent)))
				sent = []
				slen = 0
				sent.extend(id + [pad])
				slen += len(id) + 1
		return inputs


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
		# return input and attention mask
		x = np.zeros((self.n_ctx, 2), dtype=np.int32)
		m = np.zeros((self.n_ctx, self.n_ctx), dtype=np.int32)
		lx = len(self.data[index])
		x[:lx, 0] = self.data[index]
		pos = self.n_vocab + 1
		count = 0
		for idx, val in enumerate(x[:, 0]):
			if val == 0:
				pos = self.n_vocab
				m[idx - count:idx, idx - count:idx] = np.tril(np.ones((count, count)))
				count = 0
			x[idx, 1] = pos
			pos += 1
			count += 1

		return x, m

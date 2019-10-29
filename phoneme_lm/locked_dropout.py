import torch.nn as nn
import torch


class LockedDropout(nn.Module):
	def __init__(self, dropout=None):
		super(LockedDropout, self).__init__()
		self.dropout = dropout

	def forward(self, x):
		if not self.training or not self.dropout:
			return x
		# same mask for each word in a seq
		mask = x.data.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout) / (1 - self.dropout)
		mask = mask.expand_as(x)
		return mask * x


class MaskDropout(nn.Module):
	def __init__(self, dropout=None):
		super(MaskDropout, self).__init__()
		self.dropout = dropout

	def forward(self, x):
		if not self.training or not self.dropout:
			return x
		# mask out same tokens in the sequence
		# mask is of size [batch, head, seq, seq]
		mask = x.data.new_empty(x.size(0), 1, 1, x.size(3)).bernoulli_(1 - self.dropout) / (1 - self.dropout / 2)
		mask = mask.expand_as(x)
		return mask * x



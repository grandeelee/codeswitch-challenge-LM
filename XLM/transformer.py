from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
from locked_dropout import LockedDropout, MaskDropout
from embed_regularize import embedded_dropout

logger = getLogger(__name__)

def gelu(x):
	return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
	return x * torch.sigmoid(x)


ACT_FNS = {
	'relu': nn.ReLU(),
	'swish': swish,
	'gelu': gelu
}


class Conv1D(nn.Module):
	def __init__(self, dim_in, dim_out):
		"""
		expecting a 3D tensor and linearly transform the last dim to dim_out
		:param dim_in:
		:param dim_out:
		"""
		super(Conv1D, self).__init__()
		self.dim_in = dim_in
		self.dim_out = dim_out
		w = torch.empty(dim_in, dim_out)
		nn.init.normal_(w, std=0.02)
		self.w = nn.Parameter(w)
		self.b = nn.Parameter(torch.zeros(dim_out))

	def forward(self, x):  # [batch, slen, embd]
		size_out = x.size()[:-1] + (self.dim_out,)
		x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
		x = x.view(*size_out) # [batch, slen, dim_out]
		return x


class ScaledDotProductAttention(nn.Module):
	def __init__(self, n_seq, dropout=0.1, scale=True, set_mask=True):
		super(ScaledDotProductAttention, self).__init__()
		self.dropout = MaskDropout(dropout)  # nn.Dropout(dropout)  # may try use variational dropout
		self.scale = scale
		self.set_mask = set_mask
		self.register_buffer("mask", torch.tril(torch.ones(n_seq, n_seq).view(1, 1, n_seq, n_seq)))

	def forward(self, q, k, v):
		w = torch.matmul(q, k)
		if self.scale:
			w = w / math.sqrt(v.size(-1))
		if self.set_mask:
			b = self.mask[:, :, :w.size(-2), :w.size(-1)]
			w = w * b + -1e9 * (1 - b)
		w = nn.Softmax(dim=-1)(w)
		w = self.dropout(w)
		return torch.matmul(w, v)


class MultiHeadAttention(nn.Module):
	def __init__(self, dim, n_seq, cfg, scale=False):
		super(MultiHeadAttention, self).__init__()
		assert dim % cfg.n_heads == 0
		self.n_heads = cfg.n_heads
		self.split_size = dim
		self.scale = scale
		self._attn = ScaledDotProductAttention(n_seq, dropout=cfg.attn_pdrop)
		self.linear = Conv1D(dim, dim * 3)
		self.proj = Conv1D(dim, dim)
		self.proj_dropout = LockedDropout(cfg.dropouth)  #nn.Dropout(cfg.resid_pdrop)

	def merge_heads(self, x):
		x = x.permute(0, 2, 1, 3).contiguous()
		new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
		return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

	def split_heads(self, x, k=False):
		new_x_shape = x.size()[:-1] + (self.n_heads, x.size(-1) // self.n_heads)
		x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
		if k:
			# batch, heads, dim, n_seq
			return x.permute(0, 2, 3, 1)
		else:
			# batch, heads, n_seq, dim
			return x.permute(0, 2, 1, 3)

	def forward(self, x):
		# linearly transform x to k q v
		x = self.linear(x)
		q, k, v = x.split(self.split_size, dim=2)
		q = self.split_heads(q)
		k = self.split_heads(k, k=True)
		v = self.split_heads(v)
		a = self._attn(q, k, v)
		a = self.merge_heads(a)
		a = self.proj(a)
		# use variational dropout? like LSTM regularization.
		a = self.proj_dropout(a)
		return a


class LayerNorm(nn.Module):
	"Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

	def __init__(self, n_state, e=1e-5):
		super(LayerNorm, self).__init__()
		self.g = nn.Parameter(torch.ones(n_state))
		self.b = nn.Parameter(torch.zeros(n_state))
		self.e = e

	def forward(self, x):
		u = x.mean(-1, keepdim=True)
		s = (x - u).pow(2).mean(-1, keepdim=True)
		x = (x - u) / torch.sqrt(s + self.e)
		return self.g * x + self.b


class MLP(nn.Module):
	def __init__(self, n_state, cfg):  # in MLP: n_state=3072 (4 * n_embd)
		super(MLP, self).__init__()
		dim = cfg.emsize
		self.c_fc = Conv1D(dim, n_state)
		self.c_proj = Conv1D(n_state, dim)
		self.act = ACT_FNS[cfg.afn]
		self.dropout = LockedDropout(cfg.dropouth)  #nn.Dropout(cfg.resid_pdrop)

	def forward(self, x):
		h = self.act(self.c_fc(x))
		h2 = self.c_proj(h)
		return self.dropout(h2)


class Block(nn.Module):
	def __init__(self, n_seq, cfg, scale=False):
		super(Block, self).__init__()
		nx = cfg.emsize
		self.attn = MultiHeadAttention(nx, n_seq, cfg, scale)
		self.ln_1 = LayerNorm(nx)
		self.mlp = MLP(4 * nx, cfg)
		self.ln_2 = LayerNorm(nx)

	def forward(self, x):
		a = self.attn(x)
		n = self.ln_1(x + a)  # or self.ln_1(a) + x
		m = self.mlp(n)
		h = self.ln_2(n + m)  # or self.ln_2(m) + n
		return h


class TransformerModel(nn.Module):
	""" Transformer model """

	def __init__(self, cfg, vocab=10, n_seq=4):
		super(TransformerModel, self).__init__()
		self.use_pos = cfg.pos_embed
		self.vocab = vocab
		self.embed = nn.Embedding(vocab, cfg.emsize)
		self.pos_embed = nn.Embedding(n_seq, cfg.emsize)
		#  dropout try LSTM regularize paper
		# self.drop = nn.Dropout(cfg.embd_pdrop)
		self.drop = cfg.embd_pdrop
		block = Block(n_seq, cfg, scale=True)
		self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(cfg.nlayers)])
		self.lockdrop = LockedDropout(cfg.dropouti)
		nn.init.normal_(self.embed.weight, std=0.02)

	def forward(self, x):

		e = embedded_dropout(self.embed, x, dropout=self.drop if self.training else 0)
		# Add the position information to the input embeddings
		if self.use_pos:
			pos = torch.arange(x.size(-1), dtype=torch.long, device=x.device)
			p = embedded_dropout(self.pos_embed, pos, dropout=self.drop if self.training else 0)
			e = e + p
		h = self.lockdrop(e)
		for block in self.h:
			h = block(h)
		return h


# class LMHead(nn.Module):
# 	""" Language Model Head for the transformer """
#
# 	def __init__(self, model, cfg):
# 		super(LMHead, self).__init__()
# 		self.n_embd = cfg.emsize
# 		embed_shape = model.embed.weight.shape
# 		self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
# 		self.decoder.weight = model.embed.weight  # Tied weights
#
# 	def forward(self, h, pred_mask):
# 		# Truncated Language modeling logits (we remove the last token)
# 		h_trunc = h[pred_mask].contiguous().view(-1, self.n_embd)
# 		lm_logits = self.decoder(h_trunc)
# 		return lm_logits


class LMModel(nn.Module):
	""" Transformer with language model head only """

	def __init__(self, cfg, vocab=None, n_ctx=None, return_probs=False):
		super(LMModel, self).__init__()
		assert vocab > 0
		assert n_ctx > 0
		self.transformer = TransformerModel(cfg, vocab=vocab, n_seq=n_ctx)
		embed_shape = self.transformer.embed.weight.shape
		self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
		self.decoder.weight = self.transformer.embed.weight  # Tied weights
		self.return_probs = return_probs
		if self.return_probs:
			pos_emb_mask = torch.zeros(1, 1, vocab)
			pos_emb_mask[:, :, -n_ctx:] = -1e12
			self.register_buffer('pos_emb_mask', pos_emb_mask)

	def forward(self, x):
		h = self.transformer(x)
		lm_logits = self.decoder(h)
		if self.return_probs:
			lm_logits = F.softmax(lm_logits + self.pos_emb_mask, dim=-1)
		return lm_logits

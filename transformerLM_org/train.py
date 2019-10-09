import time
import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import hashlib
import dataset
from tqdm import tqdm
from options import get_args
from transformer import LMModel
from opt import OpenAIAdam


args = get_args()

# create log
log = open(args.model + '.log', mode='w', encoding='utf-8', errors='surrogateescape')

print('------------------------------------------------', file=log)
for key, value in vars(args).items():
	print('{} : {}'.format(key, value), file=log)
print('------------------------------------------------', file=log)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("device", device, "n_gpu", n_gpu, file=log)
log.flush()

# ============== data set preparation ======================
fn = '../data/corpus.{}.data'.format('new')
if os.path.exists(fn):
	print('Loading cached dataset...')
	corpus = torch.load(fn)
else:
	print('Producing dataset...')
	corpus = dataset.Corpus(args.data)
	torch.save(corpus, fn)

n_vocab = len(corpus.dictionary)
train, valid, test = corpus.train, corpus.valid, corpus.test
n_ctx = max([len(sent) for sent in train] +
            [len(sent) for sent in valid] +
            [len(sent) for sent in test])

if n_ctx > args.n_ctx:
	print('data sequence longer than model sequence, try to segment', file=log)
vocab = n_vocab + n_ctx

train_set = dataset.Dataset(train, n_ctx, n_vocab)
valid_set = dataset.Dataset(valid, n_ctx, n_vocab)
test_set = dataset.Dataset(test, n_ctx, n_vocab)

generator_params = {'batch_size': args.batch_size,
                    'shuffle': False,
                    'num_workers': 8}
train_generator = data.DataLoader(train_set, **generator_params)
valid_generator = data.DataLoader(valid_set, **generator_params)
test_generator = data.DataLoader(test_set, **generator_params)
# =================== end of data set preparation ============

lm_model = LMModel(args, vocab, n_ctx)
criterion = nn.CrossEntropyLoss(reduction='none')

n_batch_train = args.batch_size * max(n_gpu, 1)
n_updates_total = (len(train_set) // n_batch_train) * args.epochs
model_opt = OpenAIAdam(lm_model.parameters(),
                       lr=args.lr,
                       schedule=args.lr_schedule,
                       warmup=args.lr_warmup,
                       t_total=n_updates_total,
                       b1=args.b1,
                       b2=args.b2,
                       e=args.e,
                       l2=args.l2,
                       vector_l2=args.vector_l2,
                       max_grad_norm=args.max_grad_norm)

total_params = sum(x.size()[0] * x.size()[1]
                   if len(x.size()) > 1 else x.size()[0] for x in lm_model.parameters() if x.size())
print('Model total parameters:', total_params, file=log)
log.flush()

lm_model.to(device)
lm_model = nn.DataParallel(lm_model)


def evaluate(data_generator):
	# Turn on evaluation mode which disables dropout.
	total_loss = 0
	n_words = 0
	for x, m in data_generator:
		x = x.to(torch.long).to(device)
		m = m.to(torch.float).to(device)
		x_shifted = x[:, 1:, 0].contiguous().view(-1)
		lm_model.eval()
		model_opt.zero_grad()
		lm_logits = lm_model(x)
		lm_logits = lm_logits.view(-1, lm_logits.size(-1))
		lm_losses = criterion(lm_logits, x_shifted)
		lm_losses = lm_losses.view(x.size(0), x.size(-2) - 1)
		lm_losses = lm_losses * m[:, 1:]
		lm_losses = lm_losses.sum() / torch.sum(m[:, 1:])
		n_words += torch.sum(m[:, 1:])
		total_loss += lm_losses.data * torch.sum(m[:, 1:])

	return total_loss.item() / n_words


def run_epoch():
	total_loss = 0
	start_time = time.time()
	batch = 0
	n_words = 0
	for x, m in train_generator:
		x = x.to(torch.long).to(device)
		m = m.to(torch.float).to(device)
		lm_model.train()
		model_opt.zero_grad()
		lm_logits = lm_model(x)
		x_shifted = x[:, 1:, 0].contiguous().view(-1)
		lm_logits = lm_logits.view(-1, lm_logits.size(-1))
		lm_losses = criterion(lm_logits, x_shifted)
		lm_losses = lm_losses.view(x.size(0), x.size(-2) - 1)
		lm_losses = lm_losses * m[:, 1:]
		lm_losses = lm_losses.sum() / torch.sum(m[:, 1:])
		n_words += torch.sum(m[:, 1:])
		lm_losses.backward()
		model_opt.step()
		total_loss += lm_losses.data * torch.sum(m[:, 1:])
		if batch % args.log_interval == 0 and batch > 0:
			cur_loss = total_loss.item() / n_words
			elapsed = time.time() - start_time
			print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
			      'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
				epoch, batch, len(train_set) // (args.batch_size * n_gpu), model_opt.param_groups[0]['lr'],
				              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)),
				file=log)
			log.flush()
			total_loss = 0
			n_words = 0
			start_time = time.time()
		batch += 1


best_val_loss = []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
	for epoch in tqdm(range(1, args.epochs + 1)):
		epoch_start_time = time.time()
		run_epoch()

		val_loss = evaluate(valid_generator)
		print('-' * 89, file=log)
		print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
		      'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
			epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)), file=log)
		print('-' * 89, file=log)

		if val_loss < stored_loss:
			torch.save(lm_model.state_dict(), args.model + '.pt')
			print('Saving model (new best validation)', file=log)
			stored_loss = val_loss

		if len(best_val_loss) > 6 and val_loss > min(best_val_loss[:-5]):
			print('Early stop', file=log)
			break

		best_val_loss.append(val_loss)

		log.flush()

except KeyboardInterrupt:
	print('-' * 89, file=log)
	print('Exiting from training early', file=log)

# Load the best saved model.
lm_model.load_state_dict(torch.load(args.model + '.pt'))

# Run on test data.
test_loss = evaluate(test_generator)
print('=' * 89, file=log)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
	test_loss, math.exp(test_loss), test_loss / math.log(2)), file=log)
print('=' * 89, file=log)

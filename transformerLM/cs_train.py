import time
import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from cs_options import get_args
from logger import create_logger
from transformer import LMModel
from opt import OpenAIAdam
from cs_loader import check_data_params, load_data

args = get_args()

# ================= initialization ========================
logger = create_logger(args.model + '.log')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {} n_gpu: {}".format(device, n_gpu))

# ============== data set preparation ======================
args.batch_size = args.batch_size * max(n_gpu, 1)
check_data_params(args)
# load data
data = load_data(args)
args.log_interval = (len(data['cs']['train']) // args.batch_size) // 10
args.vocab_size = len(data['dictionary'])
assert args.max_len <= args.n_ctx, 'sequence length cannot accommodate max sent length'

logger.info('------------------------------------------------')
for key, value in vars(args).items():
    logger.info('{} : {}'.format(key, value))
logger.info('------------------------------------------------')

# iterators[('penn', 'train')]
# iterators[('penn', 'valid')]
# iterators[('penn', 'test')]
iterators = {}


def get_iterator(iter_name, data_set):
    """
    Create a new iterator for a dataset.
    """
    logger.info("Creating new training data iterator (%s) ..." % ','.join(
        [str(x) for x in [iter_name, data_set] if x is not None]))

    iterator = data[iter_name][data_set].get_iterator(
        shuffle=True,
        group_by_size=False,
    )

    iterators[(iter_name, data_set)] = iterator
    return iterator


def get_batch(iter_name, data_set):
    """
    Return a batch of sentences from a dataset.
    iter_name : causal
    :return batch, lengths
    """
    iterator = iterators.get((iter_name, data_set), None)
    if iterator is None:
        iterator = get_iterator(iter_name, data_set)
    try:
        x = next(iterator)
    except StopIteration:
        iterator = get_iterator(iter_name, data_set)
        x = next(iterator)
    return x


# =================== end of data set preparation ============

model = LMModel(args, args.vocab_size, args.n_ctx)
criterion = nn.CrossEntropyLoss(reduction='none')

n_updates_total = (len(data['cs']['train']) // args.batch_size) * args.epochs
model_opt = OpenAIAdam(model.parameters(),
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

logger.info("Model: {}".format(model))
logger.info("Number of parameters (model): %i" % sum([p.numel() for p in model.parameters() if p.requires_grad]))


model.to(device)
# model = nn.DataParallel(model)


def evaluate(generator):
    # Turn on evaluation mode which disables dropout.
    total_loss = 0
    n_words = 0
    for x, lengths in generator:
        alen = torch.arange(lengths.max(), dtype=torch.long, device=lengths.device)
        pred_mask = alen[None] < lengths[:, None] - 1
        y = x[:, 1:].masked_select(pred_mask[:, :-1])

        x = x.to(device)
        pred_mask = pred_mask.to(device)
        y = y.to(device)

        model.eval()
        model_opt.zero_grad()
        lm_logits = model(x)
        lm_logits = lm_logits[pred_mask].contiguous().view(-1, args.vocab_size)
        lm_losses = criterion(lm_logits, y)
        lm_losses = lm_losses.sum() / torch.sum(pred_mask)
        n_words += torch.sum(pred_mask)
        total_loss += lm_losses.data * torch.sum(pred_mask)

    return total_loss / n_words


def run_epoch(data_name, set_name):
    total_loss = 0
    start_time = time.time()
    batch = 0
    n_words = 0
    epoch_size = (len(data[data_name][set_name]) // args.batch_size)
    train_iterator = get_iterator(data_name, set_name)
    for x, lengths in tqdm(train_iterator, total=epoch_size, ncols=30):
        # generate batch
        # x, lengths = get_batch('penn', 'train')
        alen = torch.arange(lengths.max(), dtype=torch.long, device=lengths.device)
        # -1 minus away the bos index, target is the sent and </s>
        pred_mask = alen[None] < lengths[:, None] - 1
        # if params.context_size > 0:  # do not predict without context
        #     pred_mask[:params.context_size] = 0
        # select target to be first word until eos
        y = x[:, 1:].masked_select(pred_mask[:, :-1])
        assert pred_mask.sum().item() == y.size(0)

        x = x.to(device)
        pred_mask = pred_mask.to(device)
        y = y.to(device)

        # forward / loss
        model.train()
        model_opt.zero_grad()
        lm_logits = model(x)
        lm_logits = lm_logits[pred_mask].contiguous().view(-1, args.vocab_size)

        lm_losses = criterion(lm_logits, y)
        lm_losses = lm_losses.sum() / torch.sum(pred_mask)
        n_words += torch.sum(pred_mask)
        lm_losses.backward()
        model_opt.step()
        total_loss += lm_losses.data * torch.sum(pred_mask)
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / n_words
            elapsed = time.time() - start_time
            logger.debug('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f} |'.format(
                epoch, batch, epoch_size, model_opt.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            n_words = 0
            start_time = time.time()
        batch += 1


if __name__ == '__main__':
    best_val_loss = []
    stored_loss = 100000000

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            run_epoch('cs', 'train')
            run_epoch('cs', 'adapt')
            run_epoch('cs', 'cs')
            valid_iterator = get_iterator('cs', 'test')
            val_loss = evaluate(valid_iterator)
            logger.info('-' * 89)
            logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f} |'.format(
                epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
            logger.info('-' * 89)

            if val_loss < stored_loss:
                torch.save(model.state_dict(), args.model + '.pt')
                logger.info('Saving model (new best validation)')
                stored_loss = val_loss

            if len(best_val_loss) > 6 and val_loss > min(best_val_loss[:-5]):
                logger.info('Early stop')
                break

            best_val_loss.append(val_loss)

    except KeyboardInterrupt:
        logger.info('-' * 89)
        logger.info('Exiting from training early')

    # Load the best saved model.
    model.load_state_dict(torch.load(args.model + '.pt'))

    # Run on test data.
    test_iterator = get_iterator('cs', 'test')
    test_loss = evaluate(test_iterator)
    logger.debug('=' * 89)
    logger.debug('| End of training | test loss {:5.2f} | test ppl {:8.2f} |'.format(
        test_loss, math.exp(test_loss)))
    logger.debug('=' * 89)

import time
import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from options import get_args
from logger import create_logger
from transformer import LMModel
from opt import OpenAIAdam
from my_loader_newsplit import check_data_params, load_data

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
args.log_interval = 1000
args.vocab_size = len(data['dictionary'])
assert args.max_len + 2 <= args.n_ctx, 'sequence length cannot accommodate max sent length'
if args.bidirectional:
    args.directions = ['forward', 'backward']
else:
    args.directions = ['forward']

logger.info('------------------------------------------------')
for key, value in vars(args).items():
    logger.info('{} : {}'.format(key, value))
logger.info('------------------------------------------------')

iterators = {}


def get_iterator(iter_name, data_set, direction='forward'):
    """
    Create a new iterator for a dataset.
    """
    logger.info("Creating new training data iterator (%s) ..." % ','.join(
        [str(x) for x in [iter_name, data_set] if x is not None]))

    iterator = data[iter_name][data_set].get_iterator(
        shuffle=True,
        group_by_size=False,
        direction=direction
    )

    iterators[(iter_name, data_set, direction)] = iterator
    return iterator


def get_batch(iter_name, data_set, direction='forward'):
    """
    Return a batch of sentences from a dataset.
    iter_name : causal
    :return batch, lengths
    """
    iterator = iterators.get((iter_name, data_set, direction), None)
    if iterator is None:
        iterator = get_iterator(iter_name, data_set, direction)
    try:
        x = next(iterator)
    except StopIteration:
        iterator = get_iterator(iter_name, data_set, direction)
        x = next(iterator)
    return x


# =================== end of data set preparation ============


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


def concat_batches(x1, len1, x2, len2, pad_idx, eos_idx):
    """
    Concat batches with different languages.
    """
    lengths = len1 + len2
    slen, bs = lengths.max().item(), lengths.size(0)

    x = x1.new(bs, slen).fill_(pad_idx)
    x[:, :len1.max().item()].copy_(x1)

    for i in range(bs):
        l1 = len1[i]
        x[i, l1:l1 + len2[i]].copy_(x2[i, :len2[i]])

    assert (x == eos_idx).long().sum().item() == 2 * bs

    return x, lengths


def run_epoch():
    total_loss = 0
    start_time = time.time()
    n_words = 0
    epoch_size = args.epoch_size

    for batch in tqdm(range(epoch_size), ncols=100):
        for direction in args.directions:
            # generate batch
            x, lengths = get_batch('train', ('en', 'zh'), direction)

            # for sent in x:
            #     x_word = [data['dictionary'][i.item()] for i in sent]
            #     logger.debug('{}'.format(' '.join(x_word)))

            # x, lengths = concat_batches(x1, lengths1, x2, lengths2, args.pad_index, args.eos_index)
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

            # generate batch
            x, lengths = get_batch('train', ('zh', 'en'), direction)
            # x, lengths = concat_batches(x1, lengths1, x2, lengths2, args.pad_index, args.eos_index)
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
                epoch + 1, batch, epoch_size, model_opt.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            n_words = 0
            start_time = time.time()


def run_adapt_epoch(iter_name, data_set):
    total_loss = 0
    start_time = time.time()
    n_words = 0
    epoch_size = len(data[iter_name][data_set]) // args.batch_size + 1
    for batch in tqdm(range(epoch_size), ncols=100):
        for direction in args.directions:
            # generate batch
            x, lengths = get_batch(iter_name, data_set, direction)
            # for sent in x:
            #     x_word = [data['dictionary'][i.item()] for i in sent]
            #     logger.debug('{}'.format(' '.join(x_word)))
            # x, lengths = concat_batches(x1, lengths1, x2, lengths2, args.pad_index, args.eos_index)
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
                epoch + 1, batch, epoch_size, model_opt.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            n_words = 0
            start_time = time.time()

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(args, model, length, temperature=1000, top_k=20, top_p=0.0):
    bos_idx = data['dictionary'].word2id['<s>']
    context = torch.full((args.batch_size, 1), bos_idx, dtype=torch.long).to(device)
    with torch.no_grad():
        for _ in tqdm(range(length)):
            outputs = model(context)
            next_token_logits = outputs[:, -1, :] / temperature
            filtered_logits = F.softmax(top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p), dim=-1)
            next_token = torch.multinomial(filtered_logits, num_samples=1)
            context = torch.cat((context, next_token), dim=1)
    return context


if __name__ == '__main__':
    model = LMModel(args, args.vocab_size)
    criterion = nn.CrossEntropyLoss(reduction='none')

    n_updates_total = args.epoch_size * 2 * args.epochs * len(args.directions)
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

    best_val_loss = []
    stored_loss = 100000000
    epoch_start = 0
    if args.resume_train:
        # epoch_start = 0
        logger.info('resume training from epoch: {}'.format(epoch_start))
        logger.info('loading model from {}'.format(args.model + '_train.pt'))
        model.load_state_dict(torch.load(args.model + '_train.pt'))
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(epoch_start, args.epochs):
            epoch_start_time = time.time()
            if args.attn_forcing == 'decreasing':
                logger.info("decreasing attention weights for attention forcing")
                model.transformer.attn_weight = max(0.0, 0.5 - epoch * (0.5 / 20))
            elif args.attn_forcing == 'increasing':
                logger.info("increasing attention weights for attention forcing")
                model.transformer.attn_weight = min(1.0, 0.0 + epoch * (0.5 / 30))
            run_epoch()
            valid_iterator = get_iterator('cs', 'valid')
            model.transformer.attn_forcing = False
            val_loss = evaluate(valid_iterator)
            logger.info('-' * 89)
            logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid_en ppl {:8.2f} |'.format(
                epoch + 1, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
            logger.info('-' * 89)

            torch.save(model.state_dict(), args.model + '_train.pt')
            logger.info('Saving model')

            for temp in [0.1, 0.7, 1, 10, 100]:
                generated = sample_sequence(args, model, args.n_ctx, temperature=temp)
                generated = generated.cpu().numpy()
                generated_word = []
                for l in generated:
                    for x in l:
                        generated_word.append(data['dictionary'].id2word[x])
                    generated_word.append('\n')
                logger.debug(' '.join(generated_word))

            if val_loss < stored_loss:
                torch.save(model.state_dict(), args.model + '_valid.pt')
                logger.info('Saving model (new best validation)')
                stored_loss = val_loss

                # Run on test data.
                test_iterator = get_iterator('cs', 'test')
                test_loss = evaluate(test_iterator)
                logger.debug('=' * 89)
                logger.debug('| End of training | test loss {:5.2f} | test ppl {:8.2f} |'.format(
                    test_loss, math.exp(test_loss)))
                logger.debug('=' * 89)
                # Run on test data.
                test_iterator = get_iterator('cs', 'test_cs')
                test_loss = evaluate(test_iterator)
                logger.debug('=' * 89)
                logger.debug('| End of training | test_cs loss {:5.2f} | test ppl {:8.2f} |'.format(
                    test_loss, math.exp(test_loss)))
                logger.debug('=' * 89)
                # Run on test data.
                test_iterator = get_iterator('cs', 'test_en')
                test_loss = evaluate(test_iterator)
                logger.debug('=' * 89)
                logger.debug('| End of training | test_en loss {:5.2f} | test ppl {:8.2f} |'.format(
                    test_loss, math.exp(test_loss)))
                logger.debug('=' * 89)
                # Run on test data.
                test_iterator = get_iterator('cs', 'test_zh')
                test_loss = evaluate(test_iterator)
                logger.debug('=' * 89)
                logger.debug('| End of training | test_zh loss {:5.2f} | test ppl {:8.2f} |'.format(
                    test_loss, math.exp(test_loss)))
                logger.debug('=' * 89)

            if len(best_val_loss) > 17 and val_loss > min(best_val_loss[:-5]):
                logger.info('Early stop')
                break

            best_val_loss.append(val_loss)

        # adaptation
        model.load_state_dict(torch.load(args.model + '_valid.pt'))
        n_updates_total = (len(data['cs']['adapt']) // args.batch_size) * 7
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
        best_val_loss = []
        model.transformer.attn_forcing = False
        for epoch in range(7):
            epoch_start_time = time.time()
            run_adapt_epoch('cs', 'adapt')
            valid_iterator = get_iterator('cs', 'valid')
            val_loss = evaluate(valid_iterator)
            logger.info('-' * 89)
            logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f} |'.format(
                epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
            logger.info('-' * 89)

            for temp in [0.1, 0.7, 1, 10, 100]:
                generated = sample_sequence(args, model, args.n_ctx, temperature=temp)
                generated = generated.cpu().numpy()
                generated_word = []
                for l in generated:
                    for x in l:
                        generated_word.append(data['dictionary'].id2word[x])
                    generated_word.append('\n')
                logger.debug(' '.join(generated_word))

            if val_loss < stored_loss:
                torch.save(model.state_dict(), args.model + '_adapt.pt')
                logger.info('Saving model (new best validation)')
                stored_loss = val_loss
                # Run on test data.
                test_iterator = get_iterator('cs', 'test')
                test_loss = evaluate(test_iterator)
                logger.debug('=' * 89)
                logger.debug('| End of training | test loss {:5.2f} | test ppl {:8.2f} |'.format(
                    test_loss, math.exp(test_loss)))
                logger.debug('=' * 89)
                # Run on test data.
                test_iterator = get_iterator('cs', 'test_cs')
                test_loss = evaluate(test_iterator)
                logger.debug('=' * 89)
                logger.debug('| End of training | test_cs loss {:5.2f} | test ppl {:8.2f} |'.format(
                    test_loss, math.exp(test_loss)))
                logger.debug('=' * 89)
                # Run on test data.
                test_iterator = get_iterator('cs', 'test_en')
                test_loss = evaluate(test_iterator)
                logger.debug('=' * 89)
                logger.debug('| End of training | test_en loss {:5.2f} | test ppl {:8.2f} |'.format(
                    test_loss, math.exp(test_loss)))
                logger.debug('=' * 89)
                # Run on test data.
                test_iterator = get_iterator('cs', 'test_zh')
                test_loss = evaluate(test_iterator)
                logger.debug('=' * 89)
                logger.debug('| End of training | test_zh loss {:5.2f} | test ppl {:8.2f} |'.format(
                    test_loss, math.exp(test_loss)))
                logger.debug('=' * 89)

            if len(best_val_loss) > 6 and val_loss > min(best_val_loss[:-5]):
                logger.info('Early stop')
                break

            best_val_loss.append(val_loss)

    except KeyboardInterrupt:
        logger.info('-' * 89)
        logger.info('Exiting from training early')

    # Load the best saved model.
    model.load_state_dict(torch.load(args.model + '_adapt.pt'))

    # Run on test data.
    test_iterator = get_iterator('cs', 'test')
    test_loss = evaluate(test_iterator)
    logger.debug('=' * 89)
    logger.debug('| End of training | test loss {:5.2f} | test ppl {:8.2f} |'.format(
        test_loss, math.exp(test_loss)))
    logger.debug('=' * 89)

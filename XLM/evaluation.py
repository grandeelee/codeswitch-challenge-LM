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
from transformer_ori import LMModel as old_model
from transformer import LMModel
from opt import OpenAIAdam
from my_loader_newsplit import load_mono_data, check_data_params

args = get_args()
args.model = '../save/bi_directional/xlm_baseline_mix_bi_nostop_valid'
args.nlayers = 12
args.gpus = '1'
# ================= initialization ========================
logger = create_logger(args.model + '_eval.log')

# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {} n_gpu: {}".format(device, n_gpu))

# ============== data set preparation ======================
args.batch_size = 50
args.n_ctx = 70
args.max_len = 68
check_data_params(args)

# load data
data = {}
load_mono_data(args, data)
dico = data['dictionary']
args.vocab_size = len(dico)
assert args.max_len + 2 <= args.n_ctx, 'sequence length cannot accommodate max sent length'

logger.info('------------------------------------------------')
for key, value in vars(args).items():
    logger.info('{} : {}'.format(key, value))
logger.info('------------------------------------------------')

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


def getMask(dict_path):
    """
    compare vocab in dict_path with data['dictionary'], return the words in dict_path as true, else false
    :param dict_path: the dictionary path containing a list of vocab
    :return: boolen of size [vocab,]
    """
    with open(dict_path, 'r', encoding='utf-8') as f:
        target_vocab = f.read().split()
    vocab_mask = torch.empty((len(dico)), dtype=torch.bool)


def maskLogits(logits, vocab_mask):
    """
    add -1e12 to logits whose index need to be masked in calculating the vocab
    :param logits: of size [n, vocab]
    :return: size [n, vocab]
    """
    # assume already have the mask of true, false with size vocab
    assert logits.dim() == 2, 'logit dim expect 2 but not 2, cannot apply vocab mask'
    logits[:, vocab_mask] = -1e12
    return logits


def evaluate(generator):
    # Turn on evaluation mode which disables dropout.
    total_loss = 0
    n_words = 0
    with torch.no_grad():
        for x, lengths in generator:
            alen = torch.arange(lengths.max(), dtype=torch.long, device=lengths.device)
            pred_mask = alen[None] < lengths[:, None] - 1
            y = x[:, 1:].masked_select(pred_mask[:, :-1])

            x = x.to(device)
            pred_mask = pred_mask.to(device)
            y = y.to(device)
            lm_logits = model(x)
            lm_logits = lm_logits[pred_mask].contiguous().view(-1, args.vocab_size)
            lm_losses = criterion(lm_logits, y)
            lm_losses = lm_losses.sum() / torch.sum(pred_mask)
            n_words += torch.sum(pred_mask)
            total_loss += lm_losses.data * torch.sum(pred_mask)

    return total_loss / n_words


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
    bos_idx = dico.word2id['<s>']
    context = torch.full((args.batch_size, 1), bos_idx, dtype=torch.long).to(device)
    with torch.no_grad():
        for _ in tqdm(range(length)):
            outputs = model(context)
            next_token_logits = outputs[:, -1, :] / temperature
            filtered_logits = F.softmax(top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p), dim=-1)
            next_token = torch.multinomial(filtered_logits, num_samples=1)
            context = torch.cat((context, next_token), dim=1)
    return context

def write(words, m, file):
    """
    take in a list of words and m is numpy array
    :param words: list
    :param m: numpy array
    :param file: path
    :return: none
    """
    assert len(words) == m.shape[0]
    f = open(file, 'w', encoding='utf-8')
    print('%d %d' % m.shape, file=f)
    for word, m in zip(words, m):
        print(word + ' ' + ' '.join(['%.6g' % x for x in m]), file=f)
    f.close()


if __name__ == '__main__':
    model = old_model(args, args.vocab_size, args.n_ctx)
    criterion = nn.CrossEntropyLoss(reduction='none')

    logger.info("Model: {}".format(model))
    logger.info("Number of parameters (model): %i" % sum([p.numel() for p in model.parameters() if p.requires_grad]))
    model.load_state_dict(torch.load(args.model + '.pt'))
    model.to(device)
    model.eval()
    #==================== PPL ========================================
    # Run on test data.
    # test_iterator = get_iterator('cs', 'test')
    # test_loss = evaluate(test_iterator)
    # logger.debug('=' * 89)
    # logger.debug('| End of training | test loss {:5.2f} | test ppl {:8.2f} |'.format(
    #     test_loss, math.exp(test_loss)))
    # logger.debug('=' * 89)
    # # Run on test data.
    # test_iterator = get_iterator('cs', 'test_cs')
    # test_loss = evaluate(test_iterator)
    # logger.debug('=' * 89)
    # logger.debug('| End of training | test_cs loss {:5.2f} | test ppl {:8.2f} |'.format(
    #     test_loss, math.exp(test_loss)))
    # logger.debug('=' * 89)
    # # Run on test data.
    # test_iterator = get_iterator('cs', 'test_en')
    # test_loss = evaluate(test_iterator)
    # logger.debug('=' * 89)
    # logger.debug('| End of training | test_en loss {:5.2f} | test ppl {:8.2f} |'.format(
    #     test_loss, math.exp(test_loss)))
    # logger.debug('=' * 89)
    # # Run on test data.
    # test_iterator = get_iterator('cs', 'test_zh')
    # test_loss = evaluate(test_iterator)
    # logger.debug('=' * 89)
    # logger.debug('| End of training | test_zh loss {:5.2f} | test ppl {:8.2f} |'.format(
    #     test_loss, math.exp(test_loss)))
    # logger.debug('=' * 89)

    #================= Generator ====================================
    for temp in [0.7, 1, 10, 100]:
        generated = sample_sequence(args, model, args.n_ctx, temperature=temp)
        generated = generated.cpu().numpy()
        generated_word = []
        for l in generated:
            for x in l:
                generated_word.append(dico.id2word[x])
            generated_word.append('\n')
        logger.debug(' '.join(generated_word))

    #================== BLI ========================================
    matrix = model.transformer.embed.weight.data.cpu().numpy()
    if not os.path.exists(args.model + '_embed'):
        write(data['dictionary'].id2word.values(), matrix, args.model + '_embed')
    logger.info('Embed saved to {}'.format(args.model + '_embed'))

    #================== CS normalization ============================
    # use cs test
    # TODO 1) explain pos embed and why it is not nec, compare to rnn using the autoregressive obj.
    # 2) lit review of CS paper.
    # 3) implement mask
    # 4) WER, normalization.
    # 5) BLI.

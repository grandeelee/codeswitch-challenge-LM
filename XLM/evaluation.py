import time
import os
from collections import Counter, OrderedDict
import random
import math
import numpy as np
import re
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
args.nlayers = 12
args.gpus = '2'
# ================= initialization ========================
# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
# logger.info("device: {} n_gpu: {}".format(device, n_gpu))


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

def getDict(text_path):
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read().lower().split()
    cnt = Counter(text)
    ordered_cnt = OrderedDict(sorted(cnt.items(), key=lambda item: (-item[1], item[0])))
    id2word = [k for k in ordered_cnt.keys()]
    with open(text_path + '_vocab', 'w', encoding='utf-8') as f:
        f.writelines(i + '\n' for i in id2word)


def getMask(dict_path, args):
    """
    compare vocab in dict_path with data['dictionary'], return the words in dict_path as true, else false
    :param dict_path: the dictionary path containing a list of vocab
    :return: boolen of size [vocab,]
    """
    with open(dict_path, 'r', encoding='utf-8') as f:
        target_vocab = f.read().split()
    vocab_mask = torch.empty((len(dico)), dtype=torch.bool).fill_(False)
    for i, word in dico.id2word.items():
        if i in [0, 1, 2, 3]:
            continue
        if word not in target_vocab:
            vocab_mask[i] = True
    torch.save(vocab_mask, args.model + '_mask')
    logger.info('save mask to {}'.format(args.model + '_mask'))
    return vocab_mask


def getCSMask(dico, args):
    """

    :param dico:
    :param args:
    :return: boolen of size [vocab,]
    """
    cs_mask = []
    num_mask = []
    for word in dico.word2id.keys():
        if len(re.findall(r'[a-z]+', word)) > 0:
            cs_mask.append(True)
        else:
            cs_mask.append(False)
        if len(re.findall(r'\d', word)) > 0:
            num_mask.append(True)
        else:
            num_mask.append(False)
    cs_mask[:14] = [False for _ in range(14)]
    # with open(args.model + '_cs_mask', 'w', encoding='utf-8') as f:
    #     f.writelines(str(i) + '\n' for i in cs_mask)
    return cs_mask, num_mask


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


def evaluate(model, criterion, generator, mask=None):
    # Turn on evaluation mode which disables dropout.
    total_loss = 0
    n_words = 0
    if mask is not None:
        dict_mask = torch.zeros(1, args.vocab_size)
        dict_mask[:, mask] = -1e12
    with torch.no_grad():
        for x, lengths in generator:
            alen = torch.arange(lengths.max(), dtype=torch.long, device=lengths.device)
            pred_mask = alen[None] < lengths[:, None] - 1
            if args.pos_embed or args.sin_embed:
                y = x[:, 1:, 0].masked_select(pred_mask[:, :-1])
            else:
                y = x[:, 1:].masked_select(pred_mask[:, :-1])

            x = x.to(device)
            pred_mask = pred_mask.to(device)
            y = y.to(device)
            lm_logits = model(x)
            lm_logits = lm_logits[pred_mask].contiguous().view(-1, args.vocab_size)
            if mask is not None:
                dict_mask = dict_mask.to(device)
                lm_logits = lm_logits + dict_mask
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


def sample_sequence(args, model, dico, length, temperature=1000, top_k=20, top_p=0.0, mask=None):
    """

    :param args:
    :param model:
    :param dico:
    :param length:
    :param temperature:
    :param top_k:
    :param top_p:
    :param mask: a list english token is True, chinese token is False
    :return:
    """
    batch = args.batch_size
    if mask is not None:
        batch = 1
        cs_mask, num_mask = mask
    bos_idx = dico.word2id['<s>']
    context = torch.full((batch, 1), bos_idx, dtype=torch.long).to(device)
    with torch.no_grad():
        for _ in tqdm(range(length)):
            outputs = model(context)
            next_token_logits = outputs[:, -1, :]
            if mask is not None:
                next_token_logits[0][cs_mask] = next_token_logits[0][cs_mask] * 2
                next_token_logits[0][num_mask] = next_token_logits[0][num_mask] / 5
            next_token_logits /= temperature
            filtered_logits = F.softmax(top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p), dim=-1)
            next_token = torch.multinomial(filtered_logits, num_samples=1)
            if cs_mask[next_token[0]]:
                cs_mask = [not i for i in cs_mask]
                cs_mask[:14] = [False for _ in range(14)]
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


def evaluator(args, dico):
    model = LMModel(args, args.vocab_size)
    criterion = nn.CrossEntropyLoss(reduction='none')

    logger.info("Model: {}".format(model))
    logger.info("Number of parameters (model): %i" % sum([p.numel() for p in model.parameters() if p.requires_grad]))
    model.load_state_dict(torch.load(args.model + '.pt'))
    model.to(device)
    model.eval()
    # ==================== PPL ========================================
    # Run on test data.
    test_iterator = get_iterator('cs', 'test')
    test_loss = evaluate(model, criterion, test_iterator)
    logger.debug('=' * 89)
    logger.debug('| End of training | test loss {:5.2f} | test ppl {:8.2f} |'.format(
        test_loss, math.exp(test_loss)))
    logger.debug('=' * 89)
    # Run on test data.
    test_iterator = get_iterator('cs', 'test_cs')
    test_loss = evaluate(model, criterion, test_iterator)
    logger.debug('=' * 89)
    logger.debug('| End of training | test_cs loss {:5.2f} | test ppl {:8.2f} |'.format(
        test_loss, math.exp(test_loss)))
    logger.debug('=' * 89)
    # Run on test data.
    test_iterator = get_iterator('cs', 'test_en')
    test_loss = evaluate(model, criterion, test_iterator)
    logger.debug('=' * 89)
    logger.debug('| End of training | test_en loss {:5.2f} | test ppl {:8.2f} |'.format(
        test_loss, math.exp(test_loss)))
    logger.debug('=' * 89)
    # Run on test data.
    test_iterator = get_iterator('cs', 'test_zh')
    test_loss = evaluate(model, criterion, test_iterator)
    logger.debug('=' * 89)
    logger.debug('| End of training | test_zh loss {:5.2f} | test ppl {:8.2f} |'.format(
        test_loss, math.exp(test_loss)))
    logger.debug('=' * 89)
    # here use the mask and get ppl to compare with benchmark
    if not os.path.exists(args.model + '_mask'):
        mask = getMask('/home/grandee/projects/LM/data/cs/seame.full_vocab', args)
    else:
        logger.info('loading mask from {}'.format(args.model + '_mask'))
        mask = torch.load(args.model + "_mask")
    logger.info('the new vocab is {}, {} no of vocab masked'.format(args.vocab_size - sum(mask), sum(mask)))
    test_iterator = get_iterator('cs', 'test')
    test_loss = evaluate(model, criterion, test_iterator, mask)
    logger.debug('test_loss: {}'.format(test_loss))
    logger.debug('=' * 89)
    logger.debug('| End of training | test loss {:5.2f} | test ppl {:8.2f} |'.format(
        test_loss, math.exp(test_loss)))
    logger.debug('=' * 89)
    # Run on test data.
    test_iterator = get_iterator('cs', 'test_cs')
    test_loss = evaluate(model, criterion, test_iterator, mask)
    logger.debug('=' * 89)
    logger.debug('| End of training | test_cs loss {:5.2f} | test ppl {:8.2f} |'.format(
        test_loss, math.exp(test_loss)))
    logger.debug('=' * 89)
    # Run on test data.
    test_iterator = get_iterator('cs', 'test_en')
    test_loss = evaluate(model, criterion, test_iterator, mask)
    logger.debug('=' * 89)
    logger.debug('| End of training | test_en loss {:5.2f} | test ppl {:8.2f} |'.format(
        test_loss, math.exp(test_loss)))
    logger.debug('=' * 89)
    # Run on test data.
    test_iterator = get_iterator('cs', 'test_zh')
    test_loss = evaluate(model, criterion, test_iterator, mask)
    logger.debug('=' * 89)
    logger.debug('| End of training | test_zh loss {:5.2f} | test ppl {:8.2f} |'.format(
        test_loss, math.exp(test_loss)))
    logger.debug('=' * 89)

    # ================= Generator ====================================
    cs_mask = getCSMask(dico, args)
    for temp in [0.7, 1, 10, 100]:
        generated = sample_sequence(args, model, dico, args.n_ctx, temperature=temp, mask=cs_mask)
        generated = generated.cpu().numpy()
        generated_word = []
        for l in generated:
            for x in l:
                generated_word.append(dico.id2word[x])
            generated_word.append('\n')
        logger.debug(' '.join(generated_word))

    # # ================== BLI ========================================
    matrix = model.transformer.embed.weight.data.cpu().numpy()
    if not os.path.exists(args.model + '_embed'):
        write(data['dictionary'].id2word.values(), matrix, args.model + '_embed')
    logger.info('Embed saved to {}'.format(args.model + '_embed'))

    # ================== CS normalization ============================
    # use cs test
    # TODO 1) explain pos embed and why it is not nec, compare to rnn using the autoregressive obj.
    # 2) lit review of CS paper.
    # 4) WER, normalization.


if __name__ == '__main__':
    model_paths = ['/home/grandee/projects/LM/save/mono_only_train_adapt']
    # '/home/grandee/projects/LM/save/xlm_baseline_mix_adapt']
    for path in model_paths:
        args.model = path
        logger = create_logger(args.model + '_eval.log')
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

        evaluator(args, dico)


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
from my_loader import load_mono_data

args = get_args()
args.data = '../data/'
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
# args.n_ctx = 35
# args.max_len = 33
args.mono_dataset = {
    splt: os.path.join(args.data, '{}.pth'.format(splt))
    for splt in ['test']  #
}

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

model = LMModel(args, args.vocab_size, args.n_ctx)
criterion = nn.CrossEntropyLoss(reduction='none')

logger.info("Model: {}".format(model))
logger.info("Number of parameters (model): %i" % sum([p.numel() for p in model.parameters() if p.requires_grad]))
model.load_state_dict(torch.load(args.model + '.pt', map_location='cpu'))
model.to(device)
model.eval()


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


def sample_sequence(args, model, length, temperature=1.0, top_k=20, top_p=0.0, device='cpu'):
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


if __name__ == '__main__':
    # epoch_start_time = time.time()
    # valid_iterator = get_iterator('cs', 'test')
    # val_loss = evaluate(valid_iterator)
    # logger.info('-' * 89)
    # logger.info('| end of evaluation | time: {:5.2f}s | valid loss {:5.2f} | '
    #             'valid_en ppl {:8.2f} |'.format(
    #     (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
    # logger.info('-' * 89)

    generated = sample_sequence(args, model, args.n_ctx)
    generated = generated.numpy()
    generated_word = []
    for l in generated:
        for x in l:
            generated_word.append(dico.id2word[x])
        generated_word.append('\n')
    logger.debug(' '.join(generated_word))

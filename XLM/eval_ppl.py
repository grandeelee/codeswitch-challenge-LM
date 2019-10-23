import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
from options import get_args
from logger import create_logger
from transformer import LMModel
from my_loader_eval import check_data_params, load_data

args = get_args()

logger = create_logger(args.model + '_eval.log')

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
        shuffle=False,
        group_by_size=False,
    )

    iterators[(iter_name, data_set)] = iterator
    return iterator


# =================== end of data set preparation ============

model = LMModel(args, args.vocab_size, args.n_ctx)
criterion = nn.CrossEntropyLoss(reduction='none')

logger.info("Model: {}".format(model))
logger.info("Number of parameters (model): %i" % sum([p.numel() for p in model.parameters() if p.requires_grad]))

model.to(device)
logger.info('loading model from {}'.format(args.model + '_adapt.pt'))
model.load_state_dict(torch.load(args.model + '_adapt.pt'))

def evaluate(generator):
    # Turn on evaluation mode which disables dropout.
    with torch.no_grad():
        for x, lengths in generator:
            alen = torch.arange(lengths.max(), dtype=torch.long, device=lengths.device)
            pred_mask = alen[None] < lengths[:, None] - 1
            y = x[:, 1:].masked_select(pred_mask[:, :-1])

            x = x.to(device)
            pred_mask = pred_mask.to(device)
            y = y.to(device)

            model.eval()
            lm_logits = model(x)
            lm_logits = lm_logits[pred_mask].contiguous().view(-1, args.vocab_size)
            lm_losses = criterion(lm_logits, y)
            lm_losses = lm_losses.split(tuple(pred_mask.sum(dim=1)))
            lm_losses = [i.sum() for i in lm_losses]
            lm_losses = torch.stack(lm_losses)
            lm_losses = lm_losses / torch.sum(pred_mask, dim=1)

            lm_losses = torch.exp(lm_losses)
            assert x.size(0) == lm_losses.size(0), 'batch size not equal to number of sentence'
            for sent, loss in zip(x, lm_losses):
                sent_word = [data['dictionary'][i.item()] for i in sent]
                logger.debug('{}\tPPL: {}'.format(' '.join(sent_word), loss.item()))


if __name__ == '__main__':
    eval_iter = get_iterator('cs', 'eval')
    val_loss = evaluate(eval_iter)
    logger.info('-' * 89)
    logger.info('Evaluation done')
    logger.info('-' * 89)

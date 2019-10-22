import argparse
import time


def get_args():
    parser = argparse.ArgumentParser(description='Transformer based Language Model')
    # directory
    parser.add_argument('--data', type=str, default='../data/cs_para/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='../save/xlm_baseline_mix', help='location of the model')

    # model
    parser.add_argument('--pos_embed', type=bool, default=False, help='whether use position embedding')
    parser.add_argument('--emsize', type=int, default=384, help='size of word embeddings')
    parser.add_argument('--n_ctx', type=int, default=70, help='sequence length')
    parser.add_argument('--n_heads', type=int, default=12)
    parser.add_argument('--nlayers', type=int, default=12, help='number of layers')
    parser.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights')

    # optimization
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--dropouti', type=float, default=0.4)
    parser.add_argument('--dropouth', type=float, default=0.15)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)

    # training
    parser.add_argument('--gpus', type=str, default='2')
    parser.add_argument('--epochs', type=int, default=200, help='upper epoch limit')
    parser.add_argument('--epoch_size', type=int, default=10000, help='epoch size')
    parser.add_argument('--batch_size', type=int, default=30, metavar='N', help='batch size')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--tokens_per_batch', type=int, default=-1)
    parser.add_argument('--resume', type=str, default='', help='path of model to resume')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use (sgd, adam)')

    # dataset
    parser.add_argument('--max_vocab', type=int, default=50000)
    parser.add_argument('--min_count', type=int, default=0)
    parser.add_argument('--max_len', type=int, default=68)

    return parser.parse_args()

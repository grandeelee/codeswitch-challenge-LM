import torch
from my_loader_newsplit import load_binarized
from options import get_args
from logger import create_logger
from transformer import LMModel

args = get_args()
logger = create_logger(args.model + '_eval.log')
data = load_binarized('/home/grandee/projects/LM/data/cs_para/test.pth', args)

model = LMModel(args, args.vocab_size)
# for each dataset compute the number of token and average CSF
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re


def token_count(infile):
    with open(infile, 'r') as f:
        text = f.read()
    # total number of token
    print("{}: #token: {}, #vocab:{}".format(infile, len(text.split()), len(set(text.split()))))


def compute_CSF(infile):
    with open(infile, 'r') as f:
        text = f.read()
    # compute CSF
    # first get word boundarys = #words - # sent
    token_count = len(text.split())
    sent_count = len(text.split('\n'))
    # make corpus contigous line by line
    text = re.sub(r' ', '', text).split('\n')
    cn_count = 0
    en_count = 0
    for line in text:
        en = re.findall(r"[0-9a-z<>']+", line)
        cn = re.findall(r"[\u4e00-\u9fff]+", line)
        cn_count += len(cn)
        en_count += len(en)
    # print("total cn_seg is {}, en_sg is {}".format(cn_count, en_count))
    print("CSF for {} is {}".format(infile, (cn_count + en_count - sent_count) / (token_count - sent_count)))


# 1, adapt
# 2, Mono
# 3, cs.train
# 4, cs.valid
# 5. cs.test
# 6. cs.test.sg/ma
# infile = ["adapt", "mono", "cs.train", "cs.valid", "cs.test", "cs.test.ma", "cs.test.sg"]
# for file_name in infile:
# 	token_count(file_name)
# infile = ["cs.train", "cs.valid", "cs.test", "cs.test.ma", "cs.test.sg"]
# for file_name in infile:
# 	compute_CSF(file_name)
infile = ["/home/grandee/projects/LM/data/cs_big/test_cs_norm"]
#          "/home/grandee/projects/LM/data/cs_big/cs.test_en",
#          "/home/grandee/projects/LM/data/cs_big/cs.test_zh",
#          "/home/grandee/projects/LM/data/cs_big/cs.train",
#          "/home/grandee/projects/LM/data/cs_big/cs.valid",
#          "/home/grandee/projects/LM/data/cs_big/cs.test",
#          "/home/grandee/projects/LM/data/cs_benchmarking/seame.full.test",
#          "/home/grandee/projects/LM/data/cs_benchmarking/seame.full.train",
#          '/home/grandee/projects/LM/data/cs_benchmarking/seame.full.valid']
for file_name in infile:
    token_count(file_name)
    compute_CSF(file_name)

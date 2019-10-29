#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


"""
Example: python data/vocab.txt data/train.txt
vocab.txt: 1stline=word, 2ndline=count
"""

import os
import sys
import re
import torch
from logger import create_logger
from my_dictionary import Dictionary


if __name__ == '__main__':

    logger = create_logger('../save/phoneme/vocab.log')
    dico = torch.load('/home/grandee/projects/LM/data/phoneme_lm/dico.pth')
    # list all datasets in the data/ directory
    data_path = '/home/grandee/projects/LM/data/phoneme_lm/1_billion'
    files = []
    for file in os.listdir(data_path):
        if file.endswith('.txt'):
            p = os.path.join(data_path, file)
            assert os.path.isfile(p)
            files.append(p)
    for data_path in files:
        # with open(data_path, 'r', encoding='utf-8') as f:
        #     text = f.read().upper()
        # # re-tokenize apostrophe
        # text = re.sub(r" 'S", r"'S", text)
        # text = re.sub(r" 'VE", r"'VE", text)
        # text = re.sub(r" 'M", r"'M", text)
        # text = re.sub(r" 'D", r"'D", text)
        # text = re.sub(r" 'T", r"'T", text)
        # text = re.sub(r" 'RE", r"'RE", text)
        # text = re.sub(r" 'LL", r"'LL", text)
        # # remove punctuation except apostrophe
        # text = re.sub(r"[^\w\d'\s]+", '', text)
        # #  or this way
        # # import string
        # # remove = string.punctuation
        # # remove = remove.replace("''", "")  # don't remove apostrophe
        # # pattern = r"[{}]".format(remove)  # create the pattern
        # # text = re.sub(pattern, "", text)
        # # remove leading, trailing and duplicate spaces
        # text = re.sub(r"^\s+|\s+$|\s+(?=\s)", "", text)
        # # remove apostrophe if in between spaces
        # text = re.sub(r" ' ", ' ', text)
        # bin_path = data_path + '.pth'
        # with open(bin_path, 'w', encoding='utf-8') as f:
        #     f.writelines(text)

        data = dico.index_data(dico, data_path, data_path + '.pth')
        logger.info("%i words (%i unique) in %i sentences." % (
            len(data['sentences']) - len(data['positions']),
            len(data['dictionary']),
            len(data['positions'])
        ))
        if len(data['unk_words']) > 0:
            logger.info("%i unknown words (%i unique), covering %.2f%% of the data." % (
                sum(data['unk_words'].values()),
                len(data['unk_words']),
                sum(data['unk_words'].values()) * 100. / (len(data['sentences']) - len(data['positions']))
            ))
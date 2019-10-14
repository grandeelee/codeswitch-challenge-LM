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

from logger import create_logger
from my_dictionary import Dictionary


if __name__ == '__main__':

    logger = create_logger('../save/cs_new_split.log')
    # list all datasets in the data/ directory
    data_path = '../data/cs_big/'
    files = []
    for file in os.listdir(data_path):
        p = os.path.join(data_path, file)
        assert os.path.isfile(p)
        files.append(p)
    # build vocab from corpus or vocab list
    dico = Dictionary()
    if os.path.isfile('../save/cs_vocab.count'):
        logger.info("reading from vocab.count")
        dico.from_vocab('../save/cs_vocab.count')
    else:
        logger.info("building vocab from corpus")
        dico.from_corpus(files)
        dico.write_vocab('../save/cs_vocab.count')
    logger.info("")

    for file in files:
        data = dico.index_data(dico, file, file + '.pth')
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
            if len(data['unk_words']) < 30:
                for w, c in sorted(data['unk_words'].items(), key=lambda x: x[1])[::-1]:
                    logger.info("%s: %i" % (w, c))

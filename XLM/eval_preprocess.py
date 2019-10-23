#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import jieba
import re
from logger import create_logger
from my_dictionary import Dictionary

if __name__ == '__main__':
    logger = create_logger('../save/eval_vocab.log')
    logger.info('segmentation')
    jieba.set_dictionary('/home/grandee/projects/LM/data/evaluate_ppl/dict.txt')
    f = open('/home/grandee/projects/LM/data/evaluate_ppl/text', 'r', encoding='utf-8')
    f_out = open('/home/grandee/projects/LM/data/evaluate_ppl/text.gl', 'w', encoding='utf-8')
    for line in f:
        seg = ' '.join(jieba.cut(line, cut_all=False, HMM=False))
        col = re.findall(r"[0-9a-z'\u4e00-\u9fff]+", seg)
        f_out.write(' '.join(col) + '\n')

    f.close()
    f_out.close()

    files = []
    data_path = '../data/evaluate_ppl/'
    for file in os.listdir(data_path):
        if file.endswith('.gl'):
            p = os.path.join(data_path, file)
            assert os.path.isfile(p)
            files.append(p)

    # build vocab from corpus or vocab list
    dico = Dictionary()
    if os.path.isfile('../save/xlm_vocab.count'):
        logger.info("reading from vocab.count")
        dico.from_vocab('../save/xlm_vocab.count')
    else:
        logger.info("building vocab from corpus")
        dico.from_corpus(files)
        dico.write_vocab('../save/xlm_vocab.count')
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

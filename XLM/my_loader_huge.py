from logging import getLogger
import os
import numpy as np
import torch

from my_dataset import Dataset, ParallelDataset
from my_dictionary import BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD

logger = getLogger(__name__)


def process_binarized(data, params):
    """
    Process a binarized dataset and log main statistics.
    1) trim max vocab and index data to unk based on new dict
    2) trim min count ...
    """
    dico = data['dictionary']
    assert ((data['sentences'].dtype == np.uint16) and (len(dico) < 1 << 16) or
            (data['sentences'].dtype == np.int32) and (1 << 16 <= len(dico) < 1 << 31))
    logger.info("%i words (%i unique) in %i sentences. %i unknown words (%i unique) covering %.2f%% of the data." % (
        len(data['sentences']) - len(data['positions']),
        len(dico), len(data['positions']),
        sum(data['unk_words'].values()), len(data['unk_words']),
        100. * sum(data['unk_words'].values()) / (len(data['sentences']) - len(data['positions']))
    ))
    if params.max_vocab != -1:
        assert params.max_vocab > 0
        logger.info("Selecting %i most frequent words ..." % params.max_vocab)
        dico.max_vocab(params.max_vocab)
        data['sentences'][data['sentences'] >= params.max_vocab] = dico.index(UNK_WORD)
        unk_count = (data['sentences'] == dico.index(UNK_WORD)).sum()
        logger.info("Now %i unknown words covering %.2f%% of the data."
                    % (unk_count, 100. * unk_count / (len(data['sentences']) - len(data['positions']))))
    if params.min_count > 0:
        logger.info("Selecting words with >= %i occurrences ..." % params.min_count)
        dico.min_count(params.min_count)
        data['sentences'][data['sentences'] >= len(dico)] = dico.index(UNK_WORD)
        unk_count = (data['sentences'] == dico.index(UNK_WORD)).sum()
        logger.info("Now %i unknown words covering %.2f%% of the data."
                    % (unk_count, 100. * unk_count / (len(data['sentences']) - len(data['positions']))))
    if (data['sentences'].dtype == np.int32) and (len(dico) < 1 << 16):
        logger.info("Less than 65536 words. Moving data from int32 to uint16 ...")
        data['sentences'] = data['sentences'].astype(np.uint16)
    return data


def load_binarized(path, params):
    """
    Load a binarized dataset.
    """
    assert path.endswith('.pth')
    assert os.path.isfile(path), path
    logger.info("Loading data from %s ..." % path)
    data = torch.load(path)
    data = process_binarized(data, params)
    return data


def set_dico_parameters(params, data, dico):
    """
    Update dictionary parameters.
    """
    if 'dictionary' in data:
        assert data['dictionary'] == dico
    else:
        data['dictionary'] = dico

    n_words = len(dico)
    bos_index = dico.index(BOS_WORD)
    eos_index = dico.index(EOS_WORD)
    pad_index = dico.index(PAD_WORD)
    unk_index = dico.index(UNK_WORD)
    mask_index = dico.index(MASK_WORD)
    if hasattr(params, 'bos_index'):
        assert params.n_words == n_words
        assert params.bos_index == bos_index
        assert params.eos_index == eos_index
        assert params.pad_index == pad_index
        assert params.unk_index == unk_index
        assert params.mask_index == mask_index
    else:
        params.n_words = n_words
        params.bos_index = bos_index
        params.eos_index = eos_index
        params.pad_index = pad_index
        params.unk_index = unk_index
        params.mask_index = mask_index


def load_mono_data(params, data):
    """
    Load monolingual data.
    returned structure
    data['dictionary']
    data['penn']
        get_iterator()
    """
    data['cs'] = {}
    for splt in ['valid', 'test', 'adapt', 'test_cs', 'test_zh', 'test_en']:
        data['cs'][splt] = {}
        # load data / update dictionary parameters / update data
        mono_data = load_binarized(params.mono_dataset[splt], params)
        set_dico_parameters(params, data, mono_data['dictionary'])

        # create batched dataset
        dataset = Dataset(mono_data['sentences'], mono_data['positions'], mono_data['dictionary'], params)

        # remove empty and too long sentences
        dataset.remove_long_sentences(params.max_len)
        dataset.remove_unk_sentences(params.unk_tol)

        data['cs'][splt] = dataset

        logger.info("")


def load_para_data(params, data):
    """
    Load parallel data.
    data['dictionary']
    data['para']
        get_iterator()
    """
    data['train'] = {}

    logger.info('============ Parallel data (%s-%s)' % ('en', 'zh'))

    for i, splt in enumerate([('en', 'zh'), ('zh', 'en')]):
        data['train'][splt] = {}

        # load binarized datasets
        path = params.para_dataset['train'][i]
        src_data = load_binarized(path, params)

        # update dictionary parameters
        set_dico_parameters(params, data, src_data['dictionary'])

        # create ParallelDataset en zh
        dataset = Dataset(src_data['sentences'], src_data['positions'], src_data['dictionary'], params)

        dataset.remove_long_sentences(params.max_len)
        dataset.remove_unk_sentences(params.unk_tol)

        data['train'][splt] = dataset

        logger.info("")


def check_data_params(params):
    """
    Check datasets parameters.
    fill in params.mono_dataset and para_dataset params
    """
    # data path
    assert os.path.isdir(params.data), params.data

    # check monolingual datasets
    params.mono_dataset = {
        splt: os.path.join(params.data, '{}.pth'.format(splt))
        for splt in ['valid', 'test', 'train', 'test_cs', 'test_zh', 'test_en']  #
    }

    for p in params.mono_dataset.values():
        if not os.path.isfile(p):
            logger.error(f"{p} not found")
    assert all([os.path.isfile(p) for p in params.mono_dataset.values()])

    # check parallel datasets
    params.para_dataset = {
        'train': (os.path.join(params.data, 'para.en-zh.pth'),
                  os.path.join(params.data, 'para.zh-en.pth'))
    }
    for p1, p2 in params.para_dataset.values():
        if not os.path.isfile(p1):
            logger.error(f"{p1} not found")
        if not os.path.isfile(p2):
            logger.error(f"{p2} not found")
    assert all([os.path.isfile(p1) and os.path.isfile(p2) for p1, p2 in params.para_dataset.values()])


def load_data(params):
    """
    data['dictionary']
    data['penn']
            ['train']['valid']['test']
                get_iterator()
    """
    data = {}

    # monolingual datasets
    load_mono_data(params, data)
    load_para_data(params, data)

    # monolingual data summary
    logger.info('============ Data summary')
    for data_set in ['valid', 'test', 'train', 'test_cs', 'test_en', 'test_zh']:
        logger.info(
            '{: <18} - {: >5} - {: >12}:{: >10}'.format('Monolingual data', data_set, 'cs',
                                                        len(data['cs'][data_set])))
    # parallel data summary
    for (src, tgt) in data['train'].keys():
        logger.info('{: <18} - {: >5} - {: >12}:{: >10}'.format('Parallel data', 'train', '%s-%s' % (src, tgt),
                                                                len(data['train'][(src, tgt)])))

    logger.info("")
    return data

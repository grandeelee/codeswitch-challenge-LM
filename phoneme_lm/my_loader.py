from logging import getLogger
import os
import numpy as np
import torch

from my_dataset import Dataset
from my_dictionary import BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD

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
    if 'dictionary' not in data:
        data['dictionary'] = dico

    n_words = len(dico)
    bos_index = dico[BOS_WORD]
    eos_index = dico[EOS_WORD]
    pad_index = dico[PAD_WORD]
    unk_index = dico[UNK_WORD]
    if hasattr(params, 'bos_index'):
        assert params.n_words == n_words
        assert params.bos_index == bos_index
        assert params.eos_index == eos_index
        assert params.pad_index == pad_index
        assert params.unk_index == unk_index
    else:
        params.n_words = n_words
        params.bos_index = bos_index
        params.eos_index = eos_index
        params.pad_index = pad_index
        params.unk_index = unk_index


def load_mono_data(params, data, name):
    """
    load the files in mono_dataset[name] to data[name]
    :param params: parameters
    :param data: a dictionary
    :param name: a string used in check_data_params
    :return:
    """
    data[name] = {}
    for splt in range(len(params.mono_dataset[name])):
        data[name][splt] = {}
        # load data / update dictionary parameters / update data
        mono_data = load_binarized(params.mono_dataset[name][splt], params)
        set_dico_parameters(params, data, mono_data['dictionary'])

        # create batched dataset
        dataset = Dataset(mono_data['sentences'], mono_data['positions'], mono_data['dictionary'], params)

        # remove empty and too long sentences
        dataset.remove_long_sentences(params.max_len)
        dataset.remove_unk_sentences()

        data[name][splt] = dataset

        logger.info("")


def check_data_params(params, data_path, name):
    """
    store all the filenames in data_path as a list under the dictionary key name if the filename end with pth
    :param params: store in parameter
    :param data_path: all the filenames end with pth
    :param name: stored as dictionary with name
    :return: None
    """
    # data path
    assert os.path.isdir(data_path), data_path
    files = []
    # check monolingual datasets
    for file in os.listdir(data_path):
        if file.endswith('.pth'):
            p = os.path.join(data_path, file)
            assert os.path.isfile(p)
            files.append(p)

    params.mono_dataset[name] = files

    for p in params.mono_dataset[name]:
        if not os.path.isfile(p):
            logger.error(f"{p} not found")
    assert all([os.path.isfile(p) for p in params.mono_dataset[name]])


def load_data(params):
    params.mono_dataset = {}
    data = {}
    check_data_params(params, '/home/grandee/projects/LM/data/phoneme_lm/1_billion', '1b_train')
    check_data_params(params, '/home/grandee/projects/LM/data/phoneme_lm/libra', 'libra')
    # monolingual datasets
    load_mono_data(params, data, '1b_train')
    load_mono_data(params, data, 'libra')

    # monolingual data summary
    logger.info('============ Data summary')
    for name in params.mono_dataset.keys():
        for data_set in range(len(params.mono_dataset[name])):
            logger.info(
                '{: <18} - {: >5} - {: >12}:{: >10}'.format('Monolingual data', data_set, name,
                                                            len(data[name][data_set])))
        logger.info("")
    return data

if __name__ == '__main__':
    data = torch.load('/home/grandee/projects/LM/data/phoneme_lm/1_billion/news.en-00008-of-00100.pth')
    print('done')
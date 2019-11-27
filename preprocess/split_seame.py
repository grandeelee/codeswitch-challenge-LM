from tqdm import tqdm
import mmap
import re
import random


def get_num_lines(file):
    """
    a helper function to get the total number of lines from file read quickly
    :param file:
    :return:
    """
    fp = open(file, 'r+')
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def split_en_zh_cs(data):
    f = open(data, 'r', encoding='utf-8')
    en = []
    zh = []
    cs = []
    for line in tqdm(f, total=get_num_lines(data)):
        en_cols = re.findall(r"[a-z']+", line)
        zh_cols = re.findall(r'[\u4e00-\u9fff]+', line)
        if len(line.split(' ')) == 0:
            continue
        if len(zh_cols) == 0:
            en.append(line)
        elif len(en_cols) == 0:
            zh.append(line)
        else:
            cs.append(line)
    f.close()
    with open(data + '_en', 'w', encoding='utf-8') as f:
        f.writelines(l for l in en)
    with open(data + '_zh', 'w', encoding='utf-8') as f:
        f.writelines(l for l in zh)
    with open(data + '_cs', 'w', encoding='utf-8') as f:
        f.writelines(l for l in cs)


def split_train_valid_test(data):
    total = get_num_lines(data)
    split = total // 20
    with open(data, 'r', encoding='utf-8') as f:
        text = f.read().split('\n')
    random.shuffle(text)

    valid = text[:split*18]
    test = text[split*18:split*19]
    train = text[split*19:]

    with open(data + '.valid', 'w', encoding='utf-8') as f:
        f.writelines(l + '\n' for l in valid)
    with open(data + '.test', 'w', encoding='utf-8') as f:
        f.writelines(l + '\n' for l in test)
    with open(data + '.train', 'w', encoding='utf-8') as f:
        f.writelines(l + '\n' for l in train)


if __name__ == '__main__':
    data = '/home/grandee/projects/LM/data/cs_benchmarking/seame.full'
    split_train_valid_test(data)
    data = '/home/grandee/projects/LM/data/cs_benchmarking/seame.full.test'
    split_en_zh_cs(data)

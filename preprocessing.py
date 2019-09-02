import os
import argparse
from collections import Counter
import numpy as np
import tensorflow as tf


parser = argparse.ArgumentParser()

parser.add_argument('--mode',choices=['nli_train','nli_eval','train','eval'],help='Choose the mode to run.', default='nli_train')

parser.add_argument('--reasoning_train_raw_fname',type=str,default='./data/train/train-full.txt')
parser.add_argument('--reasoning_dev_raw_fname',type=str,default='./data/train/dev-full.txt')
parser.add_argument('--reasoning_test_raw_fname',type=str,default='./data/train/test-only-data.txt')

parser.add_argument('--word_embed_glove_fname', type=str, default='./data/emb/glove.6B.300d.txt')
parser.add_argument('--emb_dim', type=int, default=300)

parser.add_argument('--snli_raw_path',type=str,default='./data/nli/snli_1.0/snli_1.0_{}.txt')
parser.add_argument('--snli_bin_path',type=str,default='./data/nli/snli_1.0/snli_1.0_{}.bin')

parser.add_argument('--vocab_path', type=str, default='./data/vocab.txt')
parser.add_argument("--vocab_size", type=int, default=50000)
parser.add_argument("--min_cnt", type=int, default=3)

args = parser.parse_args()

def read_reasoning_dataset(raw_path, setname):
    """
    Read reasoning(main) dataset and return by list of dict
    :param raw_path: reasoning path
    :param setname: one of train,valid,test
    :return: [{hyp:,pre,war1,war2} ...]
    """
    assert os.path.exists(raw_path)
    read_headline = False
    with open(raw_path) as f:
        for line in f:
            split_line = line.strip().split('\t')
            if split_line[0] == '#id':
                if read_headline: raise ValueError
                sent1Idx, sent2Idx, reasonIdx, claimIdx = split_line.index('warrant0'), split_line.index(
                    'warrant1'), split_line.index('reason'), split_line.index('claim')
                titleIdx, infoIdx = split_line.index('debateTitle'), split_line.index('debateInfo')
                if setname != 'test': labelIdx = split_line.index('correctLabelW0orW1')
                continue
            to_return = {
                'w0': split_line[sent1Idx],
                'w1': split_line[sent2Idx],
                'claim': split_line[claimIdx],
                'reason': split_line[reasonIdx],
                'title': split_line[titleIdx],
                'info': split_line[infoIdx]
            }
            if setname == 'test': to_return['label'] = split_line[labelIdx]
            yield to_return


def nli_dataset_generator(raw_path, setname):
    """
    Generator to read NLI dataset
    :param raw_path:
    :param setname:
    :return: generator for [label, sentence1(premise), sentence2(hypothesis)]
    """

    fullpath = raw_path.format(setname)
    assert os.path.exists(fullpath)
    read_headline = False
    with open(fullpath) as f:
        for line in f:
            split_line = line.strip().split('\t')
            if split_line[0] not in ['neutral', 'contradiction', 'entailment']:
                if read_headline: raise ValueError
                sent1Idx, sent2Idx, labelIdx = split_line.index('sentence1'), split_line.index(
                    'sentence2'), split_line.index('gold_label')
                continue
            if not (len(split_line) == 10 and setname == 'test') and not (len(split_line) == 14 and setname in ['train', 'dev']):
                print(len(split_line))
                raise ValueError("Data integrity is violated!")
            yield [split_line[labelIdx], split_line[sent1Idx], split_line[sent2Idx]]

def build_vocab(snli_gen, reason_data, vocabpath='./data/vocab.txt', min_cnt=3):
    """
    Read dataset and build vocab list.
    """
    counter = Counter()
    main

import pickle
import os
import numpy as np
from Config import MyConfig
from random import shuffle
from gensim.models import FastText


def read_reasoning_raw_file(fname, is_test=False):
    not_in_test_kname = 'correctLabelW0orW1'
    data = []
    with open(fname,'r') as f:
        ls = f.readlines()
        d_format = ls[0].strip().split('\t')
        for line in ls[1:]:
            tmp = line.strip().split('\t')
            assert len(tmp)==len(d_format)
            item = dict()
            for k,v in zip(d_format, tmp): item[k]=v
            if not_in_test_kname not in d_format: item[not_in_test_kname]=None# No label in test data
            data.append(item)
    return data


def batch_idx_mapping(word_idx, batch):
    # TODO
    """Batch's sentence into array of index"""
    """split dict into into 4(or 5) different array"""
    total_batch = []
    for item in batch:
        real_batch = []
        for sent_idx,raw_sent in enumerate(item):
            if raw_sent in ['0','1']:  # label case
                label = [1,0] if int(raw_sent)==0 else [0,1]
                real_batch.append(label)
                continue
            real_batch.append(list())
            tokens = raw_sent.split()
            for t in tokens:
                if t in word_idx: real_batch[sent_idx].append(word_idx[t])
                else: real_batch[sent_idx].append(word_idx['_UNK_'])
        total_batch.append(real_batch)
    return total_batch


def load_reasoning_data(setname):
    raw_data = read_reasoning_raw_file(MyConfig.reasoning_tdv_fname_list[MyConfig.tdv_map[setname]])
    used_k = ['warrant0', 'warrant1', 'correctLabelW0orW1', 'reason', 'claim']
    total_item = [[item[k] for k in used_k] for item in raw_data]
    return total_item


def reasoning_batch_generator(batch_size=100,epoch=1, word_idx=None):
    pivot = 0 # idx in one epoch
    total_data = load_reasoning_data('train')

    for ep in range(epoch):
        #print('---Epoch {}/{}---'.format(ep,epoch))
        batch = []
        for idx,item in enumerate(total_data):
            batch.append(item)
            if len(batch)==batch_size:
                batch = batch_idx_mapping(word_idx, batch)
                batch = split_hori(batch, word_idx)
                yield batch
                batch = [] # ignore the remaining chunk?


def split_hori(batch, word_idx):
    items = [list() for i in range(5)]
    for b in batch:
        for idx,el in enumerate(b):
            items[idx].append(el)
    for i,el in enumerate(items): items[i] = padding(el,word_idx)
    return items


def padding(batch, word_idx):
    PAD_KEY = word_idx['_PAD_']
    maxlen = max_len(batch)
    for idx,item in enumerate(batch):
        batch[idx] = item + (maxlen-len(item))*[PAD_KEY]
    return batch

def max_len(batch):
    max_len = 0
    for i in batch:
        if len(i)>max_len:max_len=len(i)
    return max_len


def reasoning_test_data_load(setname, word_idx=None):
    assert setname in ['dev','test']
    total_data = load_reasoning_data(setname)
    total_data = batch_idx_mapping(word_idx, total_data)
    total_data = split_hori(total_data,word_idx)
    return total_data


def nli_load_raw_file():
    pass


def nli_next_batch():
    pass


def load_word_embedding_table(model_type):
    """
    Return the word embedding lookup table & word-idx map
    Returns:
        vocab_matrix : numpy array, shape : [vocab_size, embedding_dimension]
        word_idx : {'word':'idx'} dictionary. 'idx' indicates the index of 'word' in vocab_matrix
    """
    assert model_type in ['GLOVE','FASTTEXT']
    if model_type == 'GLOVE':
        print("{} embedding model load....".format(model_type))
        wordmap = list()
        with open(MyConfig.word_embed_glove_binary_fname,'r',encoding='utf8') as f:
            ls = f.readlines()
            for line in ls:
                line = line.strip().split()
                wordmap.append([line[0],line[1:]])  # [name, vector]

        special_key = ['_UNK_','_PAD_']
        for k in special_key:
            # TODO : Serialize two special key vector by `pickle` library
            if os.path.exists(k+'.txt'):
                with open(k+'.txt','rb') as f:
                    kval = pickle.load(f)
            else:
                kval = np.random.normal(0, 0.0001, MyConfig.word_embed_vector_len)
                with open(k+'.txt','wb') as f:
                    pickle.dump(kval,f)
            wordmap.append([k, kval])

        vocab_matrix = np.zeros([len(wordmap), MyConfig.word_embed_vector_len])
        word_idx = dict()
        for idx,word in enumerate(wordmap):
            vec = np.array([float(val) for val in word[1]])
            vocab_matrix[idx] = vec
            word_idx[word[0]] = idx
        print('Load Finish. vocab size : {}\n'.format(len(list(word_idx.keys()))))
        return vocab_matrix, word_idx

    if model_type == 'FASTTEXT':
        pass


if __name__ == '__main__':
    embed_matrix, word_idx = load_word_embedding_table('GLOVE')
    train_data_gen = reasoning_batch_generator(100,1, word_idx)
    for a in train_data_gen:

        break

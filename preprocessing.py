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


def load_reasoning_data(setname):
    raw_data = read_reasoning_raw_file(MyConfig.reasoning_tdv_fname_list[MyConfig.tdv_map[setname]])
    used_k = ['warrant0', 'warrant1', 'correctLabelW0orW1', 'reason', 'claim']
    total_item = [[item[k] for k in used_k] for item in raw_data]
    return total_item


def reasoning_batch_generator(batch_size=100,epoch=1):
    pivot = 0 # idx in one epoch
    total_data = load_reasoning_data('train')
    shuffle(total_data)

    for ep in range(epoch):
        print('---Epoch {}/{}---'.format(ep,epoch))
        batch = []
        for idx,item in enumerate(total_data):
            batch.append(item)
            if len(batch)==batch_size:
                yield batch
                batch = [] # ignore the remaining chunk?


def reasoning_test_data_load(setname):
    assert setname in ['dev','test']
    total_data = load_reasoning_data(setname)
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
            wordmap.append([k, np.random.normal(0, 0.0001, MyConfig.word_embed_vector_len)])

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
    a,b = load_word_embedding_table("GLOVE")

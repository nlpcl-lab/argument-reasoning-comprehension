import numpy as np
from Config import MyConfig
from random import shuffle


def reasoning_load_raw_file(fname,is_test=False):
    data = []
    with open(fname,'r') as f:
        ls = f.readlines()
        d_format = ls[0].strip().split('\t')
        for line in ls[1:]:
            tmp = line.strip().split('\t')
            assert len(tmp)==len(d_format)
            item = dict()
            for k,v in zip(d_format, tmp): item[k]=v
            if 'correctLabelW0orW1' not in item: item['correctLabelW0orW1']=None
            data.append(item)
    return data


def load_data(setname):
    raw_data = reasoning_load_raw_file(MyConfig.reasoning_tdv_fname_list[MyConfig.tdv_map[setname]])
    used_k = ['warrant0', 'warrant1', 'correctLabelW0orW1', 'reason', 'claim']
    total_item = [[item[k] for k in used_k] for item in raw_data]
    return total_item


def reasoning_batch_generator(setname,batch_size=100,epoch=1):
    assert setname in ['train','dev','test']

    pivot = 0 # idx in one epoch
    total_data = load_data(setname)
    shuffle(total_data)

    for ep in range(epoch):
        print('---Epoch {}/{}---'.format(ep,epoch))
        batch = []
        for idx,item in enumerate(total_data):
            batch.append(item)
            if len(batch)==batch_size:
                yield batch
                batch = [] # ignore the remaining chunk?


def nli_load_raw_file():
    pass

def nli_next_batch():
    pass


if __name__ == '__main__':
    for i in reasoning_batch_generator('test',10,1):
        print(i)
        input()

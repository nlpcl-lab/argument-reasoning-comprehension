import struct
import os
import argparse
from collections import Counter
import re
from tensorflow.core.example import example_pb2
from stanfordcorenlp import StanfordCoreNLP


parser = argparse.ArgumentParser()

parser.add_argument('--mode',choices=['nli_train','nli_eval','train','eval'],help='Choose the mode to run.', default='nli_train')

parser.add_argument('--reasoning_train_raw_fname',type=str,default='./data/main/train/train-full.txt')
parser.add_argument('--reasoning_dev_raw_fname',type=str,default='./data/main/dev/dev-full.txt')
parser.add_argument('--reasoning_test_raw_fname',type=str,default='./data/main/test/test-only-data.txt')
parser.add_argument('--reasoning_bin_fname',type=str,default='./data/main/{}_binary.bin')

parser.add_argument('--corenlp_path', type=str, default='./data/stanford_corenlp')

parser.add_argument('--word_embed_glove_fname', type=str, default='./data/emb/glove.6B.300d.txt')
parser.add_argument('--emb_dim', type=int, default=300)

parser.add_argument('--tokenize_strategy', choices=['corenlp','split'], default='split')
parser.add_argument('--snli_raw_path',type=str,default='./data/nli/snli_1.0/snli_1.0_{}.txt')
parser.add_argument('--snli_bin_path',type=str,default='./data/nli/snli_1.0/snli_1.0_{}.bin')

parser.add_argument('--vocab_path', type=str, default='./data/vocab.txt')
parser.add_argument("--vocab_size", type=int, default=20000)
parser.add_argument("--min_cnt", type=int, default=1)

args = parser.parse_args()

nlp = StanfordCoreNLP(args.corenlp_path)



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
                'w0': preprocess_sentence(split_line[sent1Idx]),
                'w1': preprocess_sentence(split_line[sent2Idx]),
                'claim': preprocess_sentence(split_line[claimIdx]),
                'reason': preprocess_sentence(split_line[reasonIdx]),
                'title': preprocess_sentence(split_line[titleIdx]),
                'info': preprocess_sentence(split_line[infoIdx])
            }
            if setname != 'test': to_return['label'] = split_line[labelIdx]

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
                if read_headline: continue
                print(split_line)
                print(setname)
                sent1Idx, sent2Idx, labelIdx = split_line.index('sentence1'), split_line.index(
                    'sentence2'), split_line.index('gold_label')
                read_headline = True
                continue
            yield [split_line[labelIdx], preprocess_sentence(split_line[sent1Idx]), preprocess_sentence(split_line[sent2Idx])]


def build_vocab(snli_gen, reason_data, vocabpath='./data/vocab.txt', min_cnt=3):
    """
    Read dataset and build vocab list.
    """
    if os.path.exists(vocabpath):
        with open(vocabpath, 'r') as f:
            vocab = [line.strip() for line in f.readlines()]
        return vocab

    # Step 1: Build vocab for reasoning dataset
    counter = Counter()

    for var in reason_data:
        all_data = [sent for sent in list(var.values()) if sent not in [i for i in range(10)]]
        all_token = ' '.join(all_data).split()
        assert all([isinstance(tok, str) for tok in all_token])
        counter.update(all_token)

    vocab = ['<PAD>', '<UNK>']
    vocab += [el[0] for el in counter.most_common() if el[1] >= min_cnt]
    if '<NUM>' not in vocab: vocab.insert(0, '<NUM>')

    vocab = vocab[:args.vocab_size]
    print("Reason dataset vocab size is {}".format(len(vocab)))

    # Step 1: Build vocab for NLI dataset
    counter = Counter()
    for d in snli_gen:
        data = [sent for sent in d[1:]]
        all_token = ' '.join(data).split()
        counter.update(all_token)

    nli_common_words = counter.most_common()
    while len(vocab) < args.vocab_size and len(nli_common_words) != 0:
        nli_word = nli_common_words.pop(0)[0]
        if nli_word not in vocab:
            vocab.append(nli_word)

    with open(vocabpath, 'w') as f:
        f.write('\n'.join(vocab))
    return vocab


def create_reason_binary_file(data, binfname):
    if os.path.exists(binfname): return
    with open(binfname, 'wb') as f:
        for line in data:
            example = example_pb2.Example()
            example.features.feature['w0'].bytes_list.value.extend([line['w0'].encode()])
            example.features.feature['w1'].bytes_list.value.extend([line['w1'].encode()])
            example.features.feature['claim'].bytes_list.value.extend([line['claim'].encode()])
            example.features.feature['reason'].bytes_list.value.extend([line['reason'].encode()])
            example.features.feature['title'].bytes_list.value.extend([line['title'].encode()])
            example.features.feature['info'].bytes_list.value.extend([line['info'].encode()])
            if 'label' in line:
                example.features.feature['label'].bytes_list.value.extend([line['label'].encode()])
            example_str = example.SerializeToString()
            str_len = len(example_str)
            f.write(struct.pack('q', str_len))
            f.write(struct.pack('%ds' % str_len, example_str))


def create_nli_binary_file(data_gen, binfname):
    if os.path.exists(binfname): return
    with open(binfname, 'wb') as f:
        for line in data_gen:
            example = example_pb2.Example()
            example.features.feature['premise'].bytes_list.value.extend([line[1].encode()])
            example.features.feature['hypothesis'].bytes_list.value.extend([line[2].encode()])
            example.features.feature['label'].bytes_list.value.extend([line[0].encode()])
            example_str = example.SerializeToString()
            str_len = len(example_str)
            f.write(struct.pack('q', str_len))
            f.write(struct.pack('%ds' % str_len, example_str))


def preprocess_sentence(sent):
    """
    Preprocess the raw sentence. The step of processing is like below.
    1. Tokenize using CoreNLP

    :param sent: (str)
    :return: preprocessed sentence (str)
    """

    if args.tokenize_strategy == 'corenlp':
        tokens = [tok.lower() for tok in nlp.word_tokenize(sent)]
    elif args.tokenize_strategy == 'split':
        tokens = [tok.lower() for tok in sent.split()]
    else:
        raise ValueError

    new_tokens = []

    for tok in tokens:
        match = re.match(r"([0-9]+)([a-z]+)", tok, re.I)
        if match is not None:
            items = match.groups()
            new_tokens.extend(list(items))
        else:
            new_tokens.append(tok)
    assert all([isinstance(tok, str) for tok in new_tokens])

    for idx,tok in enumerate(new_tokens):
        try:
            tok = int(tok)
            tok = '<NUM>'
        except:
            pass
        new_tokens[idx] = tok
    return ' '.join(new_tokens)


if __name__ == '__main__':
    setlist = ['train','dev','test']
    """
    Read dataset and preprocessing
    """
    reason_train, reason_dev, reason_test = read_reasoning_dataset(args.reasoning_train_raw_fname,'train'), read_reasoning_dataset(
        args.reasoning_dev_raw_fname, 'dev'), read_reasoning_dataset(args.reasoning_test_raw_fname, 'test')

    nli_train_gen = nli_dataset_generator(args.snli_raw_path, 'train')

    """
    Build vocab.
    """
    build_vocab(nli_train_gen, reason_train, args.vocab_path, args.min_cnt)

    nli_train_gen, nli_dev_gen, nli_test_gen = nli_dataset_generator(args.snli_raw_path,
                                                                     'train'), nli_dataset_generator(args.snli_raw_path,
                                                                                                     'dev'), \
                                               nli_dataset_generator(args.snli_raw_path, 'test')

    """
    Create binary file for  training and inference.
    """
    create_nli_binary_file(nli_train_gen, args.snli_bin_path.format('train'))
    create_nli_binary_file(nli_dev_gen, args.snli_bin_path.format('dev'))
    create_nli_binary_file(nli_test_gen, args.snli_bin_path.format('test'))

    create_reason_binary_file(reason_train, args.reasoning_bin_fname.format('train'))
    create_reason_binary_file(reason_dev, args.reasoning_bin_fname.format('dev'))
    create_reason_binary_file(reason_test, args.reasoning_bin_fname.format('test'))

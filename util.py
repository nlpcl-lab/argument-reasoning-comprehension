import os
import numpy as np
import codecs
import tensorflow as tf

class Vocab():
    def __init__(self, path='data/vocab.txt'):
        self.word2id, self.id2word = {}, {}
        self.vocabpath = path
        self.read_voca()

    def read_voca(self):
        assert os.path.exists(self.vocabpath)
        with open(self.vocabpath, 'r', encoding='utf8') as f:
            ls = [line.strip() for line in f.readlines()]
        for idx, word in enumerate(ls):
            self.word2id[word] = idx
            self.id2word[idx] = word
        self.unk_id = self.word2id['<UNK>']
        self.num_id = self.word2id['<NUM>']
        self.pad_id = self.word2id['<PAD>']
        self.words = list(self.word2id.keys())
        self.word_sorted = ls

    def text2ids(self, toks):
        assert isinstance(toks, list) and all([isinstance(tok, str) for tok in toks])
        ids = [self.word2id[tok] if tok in self.word2id else self.unk_id for tok in toks]
        return ids

def make_custom_embedding_matrix(vocab, hps):
    if os.path.exists(hps.custom_embed_path + '.npy'):
        mat = np.load(hps.custom_embed_path + '.npy')
        return tf.Variable(mat, trainable=True, name='embed_matrix')

    emb_mat = load_pretrain_word_embedding_matrix(hps, vocab)
    emb_mat = np.array(emb_mat, dtype=np.float32)
    np.save(hps.custom_embed_path, emb_mat)
    assert os.path.exists(hps.custom_embed_path + '.npy')

    emb_mat = tf.Variable(emb_mat, trainable=True, name='embed_matrix')

    return emb_mat


def load_pretrain_word_embedding_matrix(hps, vocab):
    """
    Return the pretrained glove matrix
    """
    assert os.path.exists(hps.embed_path)
    words = {k:None for k in vocab.words}

    with codecs.getreader('utf-8')(tf.gfile.GFile(hps.embed_path, 'rb')) as f:
        for line in f:
            line = line.strip().split()
            if line[0] in vocab.words:
                vec = [float(var) for var in line[1:]]
                assert len(vec) == hps.embed_dim
                words[line[0]] = vec

    unk_cnt = 0
    for word, vec in words.items():
        if vec is None:
            vec = np.random.normal(size=hps.embed_dim)
            words[word] = vec
            unk_cnt += 1

    print("UNK is {}".format(unk_cnt))
    assert len(list(words.keys())) == len(vocab.words)
    matrix = []
    for word in vocab.words:
        matrix.append(words[word])
    return matrix


def load_ckpt(args, saver, sess, ckpt_dir='train', ckpt_id=None):
    while True:
        if ckpt_id is None or ckpt_id == -1:
            ckpt_dir = os.path.join(args.model_path, ckpt_dir)
            ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=None)
            print(ckpt_dir)
            ckpt_path = ckpt_state.model_checkpoint_path
            print("CKPT_PATH: {}".format(ckpt_path))
            # print_tensors_in_checkpoint_file(file_name=ckpt_path, tensor_name='', all_tensors=False)
            saver.restore(sess, ckpt_path)
            return ckpt_path

def gpu_config():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    return config


def assign_specific_gpu(gpu_nums='-1'):
    assert gpu_nums is not None and gpu_nums != '-1'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_nums


def get_pretrain_weights(path):
    """ Load pretrain weights and save """
    with tf.Session(config=gpu_config()) as sess:
        ckpt_name = tf.train.latest_checkpoint(path)
        meta_name = ckpt_name + '.meta'
        save_graph = tf.train.import_meta_graph(meta_name)
        save_graph.restore(sess, ckpt_name)
        var_name_list = [var.name for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
        vardict = dict()
        for var in var_name_list:
            vardict[var] = sess.run(tf.get_default_graph().get_tensor_by_name(var))
    tf.reset_default_graph()
    return vardict

def assign_pretrain_weights(pretrain_vardicts):
    assign_op, uninitialized_varlist = [], []
    all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_op_names = []

    """
    Pretrain Values:
    esim/embed_matrix:0
    esim/bidirectional_rnn/fw/basic_lstm_cell/kernel:0
    esim/bidirectional_rnn/fw/basic_lstm_cell/bias:0
    esim/bidirectional_rnn/bw/basic_lstm_cell/kernel:0
    esim/bidirectional_rnn/bw/basic_lstm_cell/bias:0
    esim/inference_composition/bidirectional_rnn/fw/basic_lstm_cell/kernel:0
    esim/inference_composition/bidirectional_rnn/fw/basic_lstm_cell/bias:0
    esim/inference_composition/bidirectional_rnn/bw/basic_lstm_cell/kernel:0
    esim/inference_composition/bidirectional_rnn/bw/basic_lstm_cell/bias:0
    esim/prediction_layer/dense/kernel:0
    esim/prediction_layer/dense/bias:0
    esim/prediction_layer/dense_1/kernel:0
    esim/prediction_layer/dense_1/bias:0
    """

    for var in all_variables:
        varname = var.name
        new_model_var = tf.get_default_graph().get_tensor_by_name(varname)

        if varname in pretrain_vardicts:
            print('[&&]', varname)
            assign_op.append(tf.assign(new_model_var, pretrain_vardicts[varname]))
            assign_op_names.append(varname)
        else:
            if 'esim' in varname:
                print('[*]', varname)
                raise ValueError
            uninitialized_varlist.append(var)

    return assign_op, uninitialized_varlist
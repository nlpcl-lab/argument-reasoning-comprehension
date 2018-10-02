import numpy as np
import tensorflow as tf
import pickle
from Config import MyConfig
from preprocessing import load_word_embedding_table


class Model():
    def __init__(self):
        embed_matrix, self.word_idx = load_word_embedding_table('GLOVE')
        self.word_embedding = tf.Variable(embed_matrix,trainable=False,dtype=tf.float32,name='word_embedding')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self._build_model()
        self.saver = tf.train.Saver(tf.global_variables())

    def _build_model(self):
        claim = tf.placeholder(tf.float32, [None, None, MyConfig.word_embed_vector_len],name='claim_input')
        reason = tf.placeholder(tf.float32, [None, None, MyConfig.word_embed_vector_len],name='reason_input')
        w0 = tf.placeholder(tf.float32, [None, None, MyConfig.word_embed_vector_len],name='W0_input')
        w1 = tf.placeholder(tf.float32, [None, None, MyConfig.word_embed_vector_len],name='W1_input')
        label = tf.placeholder(tf.float32, [None, 2])

    def _build_op(self):
        pass

    def write_summary(self):
        pass

    def _build_cells(self):
        pass

    def _import_nli_embed_model(self):
        pass

    def train(self, sess, claim, reason, w0, w1, label):
        pass

    def test(self, sess, claim, reason, w0, w1):
        pass

    def predict(self, sess, claim, reason, w0, w1):
        pass


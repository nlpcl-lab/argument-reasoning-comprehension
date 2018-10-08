import numpy as np
import tensorflow as tf
import pickle
from Config import MyConfig
from preprocessing import load_word_embedding_table


class Model():
    def __init__(self, embed_matrix):
        self.word_embedding = tf.Variable(embed_matrix,trainable=False,dtype=tf.float32,name='word_embedding')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self._build_model()
        self.saver = tf.train.Saver(tf.global_variables())

        #Dropout rate
        self.rnn_keeprate = tf.placeholder_with_default(0.5, tf.to_float, name='rnn_keep_rate')
        self.fcn_keeprate = tf.placeholder_with_default(0.5, tf.to_float, name='rnn_keep_rate')

    def _build_model(self):
        # TODO: Make another input receiving function.
        claim = tf.placeholder(tf.float32, [None, None, MyConfig.word_embed_vector_len],name='claim_input')
        reason = tf.placeholder(tf.float32, [None, None, MyConfig.word_embed_vector_len],name='reason_input')
        w0 = tf.placeholder(tf.float32, [None, None, MyConfig.word_embed_vector_len],name='W0_input')
        w1 = tf.placeholder(tf.float32, [None, None, MyConfig.word_embed_vector_len],name='W1_input')
        label = tf.placeholder(tf.float32, [None, 2],name='Label')

        claim_enc_fw, claim_enc_bw, claim_enc_fw, claim_enc_bw, claim_enc_fw, claim_enc_bw, claim_enc_fw, claim_enc_bw = self._build_cells(self.rnn_keeprate)
        self._build_cells(self.rnn_keeprate)
        with tf.variable_scope('ciam_enc'):
            claim_outputs, claim_states = tf.nn.bidirectional_dynamic_rnn(claim_enc_fw, claim_enc_bw, claim,
                                                                          dtype=tf.float32)
        auto commit test 

    def _build_op(self):
        pass


    def write_summary(self):
        pass


    def _build_cells(self,keep_rate):
        pass


    def _import_nli_embed_model(self):
        pass

    def train(self, sess, claim, reason, w0, w1, label):
        return sess.run([self.train_op,self.cost],feed_dict={
            'claim_input:0': claim,
            'reason_input:0': reason,
            'W0_input:0': w0,
            'W1_input:0': w1,
            'Label:0': label,
        })

    def test(self, sess, claim, reason, w0, w1):
        pass

    def predict(self, sess, claim, reason, w0, w1):
        pass


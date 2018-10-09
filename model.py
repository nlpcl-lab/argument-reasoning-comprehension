import numpy as np
import tensorflow as tf
import pickle
from Config import MyConfig
from preprocessing import load_word_embedding_table


class Model():
    def __init__(self, embed_matrix):
        self.word_embedding = tf.Variable(embed_matrix,trainable=False,dtype=tf.float32,name='word_embedding')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        #Dropout rate
        self.rnn_keeprate = tf.placeholder_with_default(1.0, shape=(), name='rnn_keep_rate')
        self.fcn_keeprate = tf.placeholder_with_default(1.0, shape=(), name='fcn_keep_rate')

        self._build_model()
        self.saver = tf.train.Saver(tf.global_variables())

    def _build_model(self):
        # TODO: Make another input receiving function.
        claim = tf.placeholder(tf.int64, [None, None],name='claim_input')
        reason = tf.placeholder(tf.int64, [None, None],name='reason_input')
        w0 = tf.placeholder(tf.int64, [None, None],name='W0_input')
        w1 = tf.placeholder(tf.int64, [None, None],name='W1_input')
        labels = tf.placeholder(tf.int64, [None, 2],name='Label')

        claim = tf.Print(claim,[claim],message='claim')

        claim_enc_fw, claim_enc_bw, reason_enc_fw, reason_enc_bw, w0_enc_fw, w0_enc_bw, w1_enc_fw, w1_enc_bw = self._build_cells(self.rnn_keeprate)
        self._build_cells(self.rnn_keeprate)
        with tf.variable_scope('ciam_enc'):
            claim_outputs, claim_states = tf.nn.bidirectional_dynamic_rnn(claim_enc_fw, claim_enc_bw, tf.nn.embedding_lookup(self.word_embedding, claim),
                                                                          dtype=tf.float32)
        with tf.variable_scope('reason_enc'):
            reason_outputs, reason_states = tf.nn.bidirectional_dynamic_rnn(reason_enc_fw, reason_enc_bw, tf.nn.embedding_lookup(self.word_embedding,reason),
                                                                          dtype=tf.float32)
        with tf.variable_scope('w0_enc'):
            w0_outputs, w0_states = tf.nn.bidirectional_dynamic_rnn(w0_enc_fw, w0_enc_bw, tf.nn.embedding_lookup(self.word_embedding, w0),
                                                                    dtype=tf.float32)
        with tf.variable_scope('w1_enc'):
            w1_outputs, w1_states = tf.nn.bidirectional_dynamic_rnn(w1_enc_fw, w1_enc_bw, tf.nn.embedding_lookup(self.word_embedding, w1),
                                                                    dtype=tf.float32)

        claim_bi = tf.concat(claim_outputs, axis=2)
        reason_bi = tf.concat(reason_outputs, axis=2)
        w0_bi = tf.concat(w0_outputs, axis=2)
        w1_bi = tf.concat(w1_outputs, axis=2)

        claim_avg = tf.reduce_mean(claim_bi, 1)
        reason_avg = tf.reduce_mean(reason_bi, 1)
        w0_avg = tf.reduce_mean(w0_bi, 1)
        w1_avg = tf.reduce_mean(w1_bi, 1)

        concat0 = tf.concat([claim_avg, reason_avg, w0_avg], axis=1)
        concat1 = tf.concat([claim_avg, reason_avg, w1_avg], axis=1)

        h0 = self._fully_connected(concat0, MyConfig.fcn_hidden, 'h0_0')
        h1 = self._fully_connected(concat1, MyConfig.fcn_hidden, 'h1_0')
        w0_prob = self._fully_connected(h0, 1, 'h0_1')
        w1_prob = self._fully_connected(h1, 1, 'h1_1')

        self.logits = tf.concat([w0_prob,w1_prob],axis=1)
        self.cost, self.train_op, self.acc = self._build_ops(self.logits, labels)

    def _fully_connected(self,input_data,output_dim, names):
        dense_layer = tf.layers.dense(input_data, output_dim, activation=None,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=MyConfig.l2_coeffi),
                                 name=names+'dense')
        drop_layer = tf.layers.dropout(dense_layer, rate=self.fcn_keeprate, name=names+'drop')
        activation_layer = tf.nn.relu(drop_layer,name=names+'relu')
        return activation_layer

    def _build_ops(self,logits,targets):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets), name='loss') + tf.losses.get_regularization_loss()
        train_op = tf.train.AdamOptimizer(learning_rate=MyConfig.lr).minimize(cost,global_step=self.global_step,name='OP')
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(targets,1),tf.argmax(logits,1)),tf.float32))

        tf.summary.scalar('acc', acc)
        tf.summary.scalar('cost',cost)
        return cost, train_op, acc

    def _build_cells(self, keep_rate):
        total_cell = [tf.nn.rnn_cell.MultiRNNCell([self._cell(keep_rate) for _ in range(MyConfig.rnn_layer)]) for i in range(8)]
        return  total_cell

    def _cell(self, keep_rate):
        cell = tf.nn.rnn_cell.BasicLSTMCell(MyConfig.rnn_hidden)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=keep_rate)
        return cell

    def write_summary(self,sess, w0, w1, label, reason, claim, write_logs=True,writer=None):
        merged = tf.summary.merge_all()
        summary = sess.run(merged, feed_dict ={
            'claim_input:0': claim,
            'reason_input:0': reason,
            'W0_input:0': w0,
            'W1_input:0': w1,
            'Label:0': label,
        })
        writer.add_summary(summary, self.global_step.eval())

    def _import_nli_embed_model(self):
        pass

    def train(self, sess, w0, w1, label, reason, claim):
        return sess.run([self.train_op,self.cost,self.acc],feed_dict={
            'claim_input:0': claim,
            'reason_input:0': reason,
            'W0_input:0': w0,
            'W1_input:0': w1,
            'Label:0': label,
            self.rnn_keeprate: MyConfig.rnn_train_keeprate,
            self.fcn_keeprate: MyConfig.fcn_train_keeprate
        })

    def test(self, sess, w0, w1, label, reason, claim, write_logs=True,writer=None):
        if write_logs:
            self.write_summary(sess, w0, w1, label, reason, claim, write_logs=True,writer=None)
        return sess.run([self.logits,self.acc],feed_dict={
            'claim_input:0': claim,
            'reason_input:0': reason,
            'W0_input:0': w0,
            'W1_input:0': w1,
            'Label:0': label})
        pass

    def predict(self, sess, claim, reason, w0, w1):
        pass


import numpy as np
import tensorflow as tf
import pickle
from util import make_custom_embedding_matrix


class Model():
    def __init__(self, vocab, hps):
        self.hps = hps
        self.vocab = vocab
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self._init_hparams()
        self._add_placeholder()
        self._build_model()

        self.saver = tf.train.Saver(tf.global_variables())
        self.summaries = tf.summary.merge_all()

    def _init_hparams(self):
        print("init hparams")
        self.batch_size = self.hps.batch_size
        self.max_seq_len = self.hps.max_enc_len
        self.vocab_size = self.hps.vocab_size
        self.emb_dim = self.hps.embed_dim
        self.hidden_dim = self.hps.main_hidden_dim
        self.esim_hidden_dim = self.hps.esim_hidden_dim
        self.fcn_hidden_dim = self.hps.main_fcn_hidden_dim

        self.lr = self.hps.learning_rate
        self.l2_coeff = self.hps.l2_coeff
        self.clip_value = self.hps.max_grad_norm

    def _add_placeholder(self):
        print("add placeholder")
        # Data batch
        self.w0_batch = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_len], name='w0_input')
        self.w1_batch = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_len], name='w1_input')
        self.label = tf.placeholder(tf.int32, [self.batch_size, 2], name='label')
        self.w0_len = tf.placeholder(tf.int32, shape=[self.batch_size], name='w0_len')
        self.w1_len = tf.placeholder(tf.int32, shape=[self.batch_size], name='w1_len')

        self.claim_batch = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_len], name='claim_input')
        self.claim_len = tf.placeholder(tf.int32, shape=[self.batch_size], name='claim_len')

        self.reason_batch = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_len], name='reason_input')
        self.reason_len = tf.placeholder(tf.int32, shape=[self.batch_size], name='reason_len')

        # Dropout rate
        self.rnn_keeprate = tf.placeholder_with_default(1.0, shape=(), name='rnn_keep_rate')
        self.fcn_keeprate = tf.placeholder_with_default(1.0, shape=(), name='fcn_keep_rate')

    def add_embedding(self):
        print('add embedding')
        self.embedding_matrix = make_custom_embedding_matrix(self.vocab, self.hps)
        self.emb_w0 = tf.nn.embedding_lookup(self.embedding_matrix, self.w0_batch)
        self.emb_w1 = tf.nn.embedding_lookup(self.embedding_matrix, self.w1_batch)
        self.emb_claim = tf.nn.embedding_lookup(self.embedding_matrix, self.claim_batch)
        self.emb_reason = tf.nn.embedding_lookup(self.embedding_matrix, self.reason_batch)

    def build_ESIM_sentence_encoding(self):
        print('add pretrain ESIM')
        with tf.variable_scope('esim'):
            fw_cell, bw_cell = self._build_cells(self.rnn_keeprate, 'input_encoding', self.esim_hidden_dim)

            claim_outputs, claim_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                                      self.emb_claim,
                                                                      sequence_length=self.claim_len,
                                                                      dtype=tf.float32)
            reason_outputs, reason_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                                            self.emb_reason,
                                                                            sequence_length=self.reason_len,
                                                                            dtype=tf.float32)
            w0_outputs, reason_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                                            self.emb_w0,
                                                                            sequence_length=self.w0_len,
                                                                            dtype=tf.float32)
            w1_outputs, reason_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                                            self.emb_w1,
                                                                            sequence_length=self.w1_len,
                                                                            dtype=tf.float32)

            claim_bi = tf.concat(claim_outputs, axis=2)
            reason_bi = tf.concat(reason_outputs, axis=2)
            w0_bi = tf.concat(w0_outputs, axis=2)
            w1_bi = tf.concat(w1_outputs, axis=2)

            enhance_c_l, enhance_w0_l = self._local_inference(claim_bi, w0_bi)
            enhance_w0_l, enhance_r_l = self._local_inference(w0_bi, reason_bi)
            enhance_w0_l, enhance_w1_l = self._local_inference(w0_bi, w1_bi)
            enhance_c_r, enhance_w1_r = self._local_inference(claim_bi, w1_bi)
            enhance_w1_r, enhance_r_r = self._local_inference(w1_bi, reason_bi)
            enhance_w1_r, enhance_w0_r = self._local_inference(w1_bi, w0_bi)

            self.cw0_encode = self._inference_composition(enhance_c_l, enhance_w0_l)
            self.rw0_encode = self._inference_composition(enhance_w0_l, enhance_r_l)
            self.w0w1_encode = self._inference_composition(enhance_w0_l, enhance_w1_l)
            self.cw1_encode = self._inference_composition(enhance_c_r, enhance_w1_r)
            self.rw1_encode = self._inference_composition(enhance_w1_r, enhance_r_r)
            self.w1w0_encode = self._inference_composition(enhance_w1_r, enhance_w0_r)

    def _local_inference(self, a, b):
        """
        Local inference using encoded input using attention, and enhance it.
        """
        with tf.variable_scope('local_inference') as scope:
            attentionweights = tf.matmul(a, tf.transpose(b, [0, 2, 1]))
            attn_a = tf.nn.softmax(attentionweights)
            attn_b = tf.transpose(tf.nn.softmax(tf.transpose(attentionweights)))

            a_hat = tf.matmul(attn_a, b)
            b_hat = tf.matmul(attn_b, a)

            a_diff = tf.subtract(a, a_hat)
            b_diff = tf.subtract(b, b_hat)

            a_matmul = tf.multiply(a, a_hat)
            b_matmul = tf.multiply(b, b_hat)

            # Below two value is equal to Equation 14 and 15, respectively.
            enhance_premise = tf.concat([a, a_hat, a_diff, a_matmul], axis=2)
            enhance_hypothesis = tf.concat([b, b_hat, b_diff, b_matmul], axis=2)

            return enhance_premise, enhance_hypothesis

    def _inference_composition(self, enhance_pre, enhance_hyp, scope='inference_composition'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
            fw_cell, bw_cell = self._build_cells(self.rnn_keeprate, scope, self.esim_hidden_dim)

            v1_outputs, v1_states = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,
                                                                    enhance_pre,
                                                                    dtype=tf.float32)

            v2_outputs, v2_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                                    enhance_hyp,
                                                                    dtype=tf.float32)
            premise_outputs = tf.concat(v1_outputs, axis=2)
            hypothesis_outputs = tf.concat(v2_outputs, axis=2)

            pre_avg = tf.reduce_mean(premise_outputs, axis=1)
            hyp_avg = tf.reduce_mean(hypothesis_outputs, axis=1)
            pre_max = tf.reduce_max(premise_outputs, axis=1)
            hyp_max = tf.reduce_max(hypothesis_outputs, axis=1)

            final_value = tf.concat([pre_avg, pre_max, hyp_avg, hyp_max], axis=1)
            return final_value

    def _build_model(self):
        print('build model')
        with tf.variable_scope("mainmodel") as scope:
            self.add_embedding()
            claim_enc_fw, claim_enc_bw = self._build_cells(self.rnn_keeprate, scope='encode_claim', hidden_dim=self.hidden_dim)
            reason_enc_fw, reason_enc_bw = self._build_cells(self.rnn_keeprate, scope='encode_reason', hidden_dim=self.hidden_dim)
            w_enc_fw, w_enc_bw = self._build_cells(self.rnn_keeprate, scope='encode_warrant', hidden_dim=self.hidden_dim)

            with tf.variable_scope('ciam_enc'):
                claim_outputs, claim_states = tf.nn.bidirectional_dynamic_rnn(claim_enc_fw, claim_enc_bw, self.emb_claim,
                                                                              dtype=tf.float32)
            with tf.variable_scope('reason_enc'):
                reason_outputs, reason_states = tf.nn.bidirectional_dynamic_rnn(reason_enc_fw, reason_enc_bw, self.emb_reason,
                                                                              dtype=tf.float32)
            with tf.variable_scope('w0_enc'):
                w0_outputs, w0_states = tf.nn.bidirectional_dynamic_rnn(w_enc_fw, w_enc_bw, self.emb_w0, dtype=tf.float32)
            with tf.variable_scope('w1_enc'):
                w1_outputs, w1_states = tf.nn.bidirectional_dynamic_rnn(w_enc_fw, w_enc_bw, self.emb_w1, dtype=tf.float32)

        self.build_ESIM_sentence_encoding()

        with tf.variable_scope("mainmodel") as scope:
            claim_bi = tf.concat(claim_outputs, axis=2)
            reason_bi = tf.concat(reason_outputs, axis=2)
            w0_bi = tf.concat(w0_outputs, axis=2)
            w1_bi = tf.concat(w1_outputs, axis=2)

            claim_avg = tf.reduce_mean(claim_bi, 1)
            reason_avg = tf.reduce_mean(reason_bi, 1)
            w0_avg = tf.reduce_mean(w0_bi, 1)
            w1_avg = tf.reduce_mean(w1_bi, 1)

            claim_max = tf.reduce_max(claim_bi, 1)
            reason_max = tf.reduce_max(reason_bi, 1)
            w0_max = tf.reduce_max(w0_bi, 1)
            w1_max = tf.reduce_max(w1_bi, 1)

            concat0 = tf.concat([self.cw0_encode, self.rw0_encode, self.w0w1_encode, claim_avg, reason_avg, w0_avg, claim_max, reason_max, w0_max], axis=1)
            concat1 = tf.concat([self.cw1_encode, self.rw1_encode, self.w1w0_encode, claim_avg, reason_avg, w1_avg, claim_max, reason_max, w1_max], axis=1)

            h0 = self._fully_connected(concat0, self.fcn_hidden_dim, 'h0_0')
            h1 = self._fully_connected(concat1, self.fcn_hidden_dim, 'h1_0')
            with tf.variable_scope('w0_prob'):
                w0_prob = self._fully_connected(h0, 1, 'h0_1', use_dropout=False, use_activation=False)
            with tf.variable_scope('w1_prob'):
                w1_prob = self._fully_connected(h1, 1, 'h0_1', use_dropout=False, use_activation=False)
            with tf.variable_scope('logits'):
                self.logits = tf.concat([w0_prob, w1_prob], axis=1, name='logits')

            self._build_ops()
            self.train_op = self.optimization_with_esim_freezing()

    def _fully_connected(self,input_data,output_dim, names, use_activation=True, use_dropout=True):
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2_coeff)

        dense_layer = tf.layers.dense(input_data, output_dim, activation=None,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 kernel_regularizer=regularizer,
                                 name=names+'dense')

        if use_dropout:
            drop_layer = tf.nn.dropout(dense_layer, keep_prob=self.fcn_keeprate, name=names+'drop')
        else:
            drop_layer = dense_layer

        if use_activation:
            activation_layer = tf.nn.relu(drop_layer, name=names + 'relu')
        else:
            activation_layer = drop_layer
        return activation_layer

    def _build_ops(self):
        self.prediction_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label), name='loss')
        tf.summary.scalar('prediction_loss', self.prediction_cost)

        self.regularization_cost = tf.losses.get_regularization_loss()

        tf.summary.scalar('regularization_loss', self.regularization_cost)

        self.cost = self.prediction_cost + self.regularization_cost

        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.label, 1), tf.argmax(self.logits, 1)), tf.float32))

        tf.summary.scalar('acc', self.acc)
        tf.summary.scalar('cost',self.cost)


    def optimization_with_esim_freezing(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "mainmodel/")
        gradients = tf.gradients(self.cost, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, self.hps.max_grad_norm)

        tf.summary.scalar('global_norm', global_norm)

        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')
        return train_op

    def _build_cells(self, keep_rate, scope, hidden_dim):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            fw_cell, bw_cell = [self._cell(keep_rate, hidden_dim) for _ in range(2)]
            return fw_cell, bw_cell

    def _cell(self, keep_rate, hidden_dim):
        cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=keep_rate)
        return cell

    def run_step(self, batch,sess, fcn_keeprate, rnn_keeprate, is_train=False):
        feeddict = self.make_feeddict(batch)

        to_return = {
            'loss': self.cost,
            'accuracy':self.acc,
            'summaries':self.summaries,
            'global_step': self.global_step
        }
        if is_train:
            to_return['train_op'] = self.train_op
            feeddict[self.fcn_keeprate] = fcn_keeprate
            feeddict[self.rnn_keeprate] = rnn_keeprate

        return sess.run(to_return, feed_dict=feeddict)

    def run_eval(self, batch, sess):
        feeddict = self.make_feeddict(batch)
        to_return = {
            'accuracy':self.acc
        }
        return sess.run(to_return, feed_dict=feeddict)

    def make_feeddict(self, batch):
        feed_dict = {}
        feed_dict[self.w0_batch] = batch.sent0_batch
        feed_dict[self.w0_len] = batch.sent0_lens
        feed_dict[self.w1_batch] = batch.sent1_batch
        feed_dict[self.w1_len] = batch.sent1_lens

        feed_dict[self.claim_batch] = batch.claim_batch
        feed_dict[self.claim_len] = batch.claim_lens

        feed_dict[self.reason_batch] = batch.reason_batch
        feed_dict[self.reason_len] = batch.reason_lens

        feed_dict[self.label] = batch.label

        return feed_dict


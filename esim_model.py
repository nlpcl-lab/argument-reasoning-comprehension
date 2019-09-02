import tensorflow as tf
from util import make_custom_embedding_matrix
'''
ESIM model for sentence pair embedding
Currently Not implemented yet.
Reference the existing ESIM Model Code, link is https://github.com/nyu-mll/multiNLI/blob/master/python/models/esim.py
'''

class ESIM:
    def __init__(self, vocab, hps):
        self.hps = hps
        self.vocab = vocab
        self._init_hparams()
        self._add_placeholder()
        self._build_model()

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.saver = tf.train.Saver(tf.global_variables())
        self.summaries = tf.summary.merge_all()

    def  _init_hparams(self):
        self.max_seq_len = self.hps.max_enc_len
        self.vocab_size = self.hps.vocab_size
        self.emb_dim = self.hps.emb_dim
        self.hidden_dim = self.hps.hidden_dim
        self.lr = self.hps.lr
        self.l2_coeff = self.hps.l2_coeff
        self.clip_value = self.hps.max_grad_norm

    def _add_placeholder(self):
        # Data batch
        self.premise_batch = tf.placeholder(tf.int32, [None, None], name='hyp_input')
        self.hypothesis_batch = tf.placeholder(tf.int32, [None, None], name='pre_input')
        self.label = tf.placeholder(tf.int32, [None, 3], name='label')
        self.premise_len = tf.placeholder(tf.int32, shape=(), name='premise_len')
        self.hypothesis_len = tf.placeholder(tf.int32, shape=(), name='hypothesis_len')
        # Dropout rate
        self.rnn_keeprate = tf.placeholder_with_default(1.0, shape=(), name='rnn_keep_rate')
        self.fcn_keeprate = tf.placeholder_with_default(1.0, shape=(), name='fcn_keep_rate')

    def add_embedding(self):
        self.embedding_matrix = make_custom_embedding_matrix(self.vocab, self.hps)
        self.emb_premise = tf.nn.embedding_lookup(self.embedding_matrix, self.premise_len)
        self.emb_hypothesis = tf.nn.embedding_lookup(self.embedding_matrix, self.hypothesis_len)

    def _build_model(self):
        with tf.variable_scope('esim'):
            self.initializer = tf.random_normal_initializer(0.0, 1e-2)

            self.add_embedding()
            # Embeded input encoding using bi-LSTM
            pre_list, hyp_list = self._input_encoding(scope='input_encoding')

            # Local inference (attention) and Enhancement if local inference information.
            enhance_pre, enhance_hyp = self._local_inference(pre_list, hyp_list)

            # Inference Composition & Pooling
            final_vector = self._inference_composition(enhance_pre, enhance_hyp, scope='inference_composition')

            # Final inference layer
            self.logits = self._fully_connected_layer(final_vector, 'prediction_layer')

            self._build_op()

    def _input_encoding(self, scope):
        fw_cell, bw_cell = self._build_cells(self.rnn_keeprate, scope)

        pre_outputs, pre_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                                  self.emb_premise,
                                                                  sequence_length=self.premise_len,
                                                                  dtype=tf.float32)
        hyp_outputs, hyp_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                                  self.emb_hypothesis,
                                                                  sequence_length=self.hypothesis_len,
                                                                  dtype=tf.float32)

        pre_bi = tf.concat(pre_outputs, axis=2)
        hyp_bi = tf.concat(hyp_outputs, axis=2)

        pre_list = tf.unstack(pre_bi, axis=1)
        hyp_list = tf.unstack(hyp_bi, axis=1)
        return pre_bi, hyp_bi

    def _local_inference(self, pre_list, hyp_list):
        """
        Local inference using encoded input using attention, and enhance it.
        :param pre_list:
        :param hyp_list:
        :param pre_len:
        :param hyp_len:
        :return:
        """
        with tf.variable_scope('local_inference') as scope:
            attentionweights = tf.matmul(pre_list, tf.transpose(hyp_list, [0, 2, 1]))
            attn_a = tf.nn.softmax(attentionweights)
            attn_b = tf.transpose(tf.nn.softmax(tf.transpose(attentionweights)))

            a_hat = tf.matmul(attn_a, hyp_list)
            b_hat = tf.matmul(attn_b, pre_list)

            a_diff = tf.subtract(pre_list, a_hat)
            b_diff = tf.subtract(hyp_list, b_hat)

            a_matmul = tf.multiply(pre_list, a_hat)
            b_matmul = tf.multiply(hyp_list, b_hat)

            # Below two value is equal to Equation 14 and 15, respectively.
            enhance_premise = tf.concat([pre_list, a_hat, a_diff, a_matmul], axis=2)
            enhance_hypothesis = tf.concat([hyp_list, b_hat, b_diff, b_matmul], axis=2)

            return enhance_premise, enhance_hypothesis

    def _inference_composition(self, enhance_pre, enhance_hyp, scope):
        with tf.variable_scope(scope) as scope:
            fw_cell, bw_cell = self._build_cells(self.rnn_keeprate, scope)

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
            pre_max = tf.reduce_mean(premise_outputs, axis=1)
            hyp_max = tf.reduce_mean(hypothesis_outputs, axis=1)

            final_value = tf.concat([pre_avg, pre_max, hyp_avg, hyp_max], axis=1)
            return final_value

    def _fully_connected_layer(self, inputs, scope):
        with tf.variable_scope(scope) as scope:
            inp1 = tf.nn.dropout(inputs, keep_prob=self.fcn_keeprate)
            inp1 = tf.layers.dense(inp1, self.hidden_dim, tf.nn.tanh, kernel_initializer=self.initializer)

            inp2 = tf.nn.dropout(inp1, keep_prob=self.fcn_keeprate)
            logits = tf.layers.dense(inp2, self.hidden_dim, None, kernel_initializer=self.initializer)
            return logits

    def _build_op(self):
        # loss
        self.prediction_cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label), name='loss')
        self.regularization_cost = tf.losses.get_regularization_loss()
        tf.summary.scalar('prediction_loss', self.prediction_cost)
        tf.summary.scalar('regularization_loss', self.regularization_cost)

        self.cost = self.prediction_cost + self.l2_coeff * self.regularization_cost

        # acc
        label_pred = tf.argmax(self.logits, 1, name='label_pred')
        label_real = tf.argmax(self.label, 1, name='label_real')
        correct = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_real, tf.int32))

        self.acc = tf.reduce_mean(tf.cast(correct, tf.float32), name='acc')
        tf.summary.scalar('acc', self.acc)

        # optimization
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        tvars = tf.trainable_variables()
        gradients = tf.gradients(self.cost, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, self.clip_value)
        tf.summary.scalar('global_norm', global_norm)

        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')

    def _build_cells(self, keep_rate, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            fw_cell, bw_cell = [self._cell(keep_rate) for _ in range(2)]
            return fw_cell, bw_cell

    def _cell(self, keep_rate):
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=keep_rate)
        return cell

    def train(self, sess, hyp, pre, label, hyp_len, pre_len):
        return sess.run([self.train_op, self.cost, self.acc, self.summaries], feed_dict={
            self.hypothesis_batch: hyp,
            self.hypothesis_len: hyp_len,
            self.premise_batch: pre,
            self.premise_len: pre_len,
            self.label: label
        })

    def test(self, sess, hyp, pre, label, hyp_len, pre_len):
        return sess.run([self.cost, self.acc, self.summaries], feed_dict={
            self.hypothesis_batch:hyp,
            self.hypothesis_len:hyp_len,
            self.premise_batch:pre,
            self.premise_len:pre_len,
            self.label:label
        })

import tensorflow as tf
from Config import MyConfig
'''
ESIM model for sentence pair embedding
'''

class ESIM():
    def __init(self, embed_matrix):
        self.word_embedding = tf.Variable(embed_matrix, trainable=False, dtype=tf.float32, name='word_embedding')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # Dropout rate
        self.rnn_keeprate = tf.placeholder_with_default(1.0, shape=(), name='rnn_keep_rate')
        self.fcn_keeprate = tf.placeholder_with_default(1.0, shape=(), name='fcn_keep_rate')

        self._build_model()
        self.saver = tf.train.Saver(tf.global_variables())

    def _build_model(self):
        pre = tf.placeholder(tf.int64, [None, None], name='hyp_input')
        hyp = tf.placeholder(tf.int64, [None, None], name='pre_input')
        label = tf.placeholder(tf.int64,[None,3],name='label')


        pre_list, hyp_list = self._input_encoding(pre,hyp)
        attention_res = self._local_inference(pre_list, hyp_list, pre, hyp)
        comp_res = self._inference_composition(attention_res)
        self._pooling_layer()
        logits = self._fully_connected_layer()
        self.cost,self.label = self._build_op(logits)

    def _input_encoding(self,pre,hyp):
        pre_enc_fw, pre_enc_bw, hyp_enc_fw, hyp_enc_bw = self._build_cells(self.rnn_keeprate)

        pre_outputs, pre_states = tf.nn.bidirectional_dynamic_rnn(pre_enc_fw, pre_enc_bw,
                                                                  tf.nn.embedding_lookup(self.word_embedding, pre),
                                                                  dtype=tf.float32)
        hyp_outputs, hyp_states = tf.nn.bidirectional_dynamic_rnn(hyp_enc_fw, hyp_enc_bw,
                                                                  tf.nn.embedding_lookup(self.word_embedding, hyp),
                                                                  dtype=tf.float32)

        pre_bi = tf.concat(pre_outputs, axis=2)
        hyp_bi = tf.concat(hyp_outputs, axis=2)

        pre_list = tf.unstack(pre_bi, axis=1)
        hyp_list = tf.unstack(hyp_bi, axis=1)
        return pre_list, hyp_list

    def _local_inference(self, pre_list, hyp_list, pre, hyp):

        score = []
        pre_att, hyp_att = [],[]
        alpha,beta = [],[]
        pass

    def _inference_composition(self):
        pass

    def _pooling_layer(self):
        pass

    def _fully_connected_layer(self):
        pass

    def _build_op(self):
        pass

    def _build_cells(self, keep_rate):
        total_cell = [tf.nn.rnn_cell.MultiRNNCell([self._cell(keep_rate) for _ in range(MyConfig.rnn_layer)]) for i in
                      range(4)]
        return total_cell

    @staticmethod
    def _cell(keep_rate):
        cell = tf.nn.rnn_cell.BasicLSTMCell(MyConfig.rnn_hidden)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=keep_rate)
        return cell

    def train(self, sess, hyp, pre, label):
        return sess.run([self.train_op, self.cost, self.acc], feed_dict={
            'hyp_input:0':hyp,
            'pre_input:0':pre,
            'label:0':label
        })

    def test(self, sess, hyp, pre, label, write_logs=True, writer=None):
        if write_logs:
            self.write_summary(sess, hyp, pre, label, write_logs=True, writer=writer)
        return sess.run([self.logits,self.acc],feed_dict={
            'hyp_input:0':hyp,
            'pre_input:0':pre,
            'label:0':label
        })

    def write_summary(self, ses, hyp, pre, label, write_logs, writer):
        pass

    
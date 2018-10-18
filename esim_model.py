import tensorflow as tf
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

        pre_state, hyp_state = self._input_encoding(pre,hyp,label)

    def _input_encoding(self,pre,hyp,label):
        pre_enc_fw, pre_enc_bw, hyp_enc_fw, hyp_enc_bw = self._build_cells(self.rnn_keeprate)

        pre_outputs, pre_states = tf.nn.bidirectional_dynamic_rnn(pre_enc_fw, pre_enc_bw,
                                                                  tf.nn.embedding_lookup(self.word_embedding, pre),
                                                                  dtype=tf.float32)
        hyp_outputs, hyp_states = tf.nn.bidirectional_dynamic_rnn(hyp_enc_fw, hyp_enc_bw,
                                                                  tf.nn.embedding_lookup(self.word_embedding, hyp),
                                                                  dtype=tf.float32)




    def _local_inference(self):
        pass

    def _inference_composition(self):
        pass

    def _build_op(self):
        pass

    def _build_cells(self):
        pass

    def _cell(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def write_summary(self):
        pass

    
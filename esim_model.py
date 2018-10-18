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
        hyp = tf.placeholder(tf.int64, [None, None], name='hyp_input')
        pre = tf.placeholder(tf.int64, [None, None], name='pre_input')
        label = tf.placeholder(tf.int64,[None,3],name='label')

    def _input_encoding(self):
        pass

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

    
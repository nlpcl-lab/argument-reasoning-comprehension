from util import option_parser
from preprocessing import reasoning_batch_generator, reasoning_test_data_load, load_word_embedding_table, encode_batch
from Config import MyConfig
import tensorflow as tf
from model import Model

'''
NeuralNet Script file
'''


def train(batch_size, epoch, word_embed):
    embed_matrix,word_idx = word_embed
    train_data_gen = reasoning_batch_generator(batch_size, epoch, word_idx)
    dev_raw = reasoning_test_data_load('dev', word_idx)
    model = Model(embed_matrix)

    with tf.Session() as sess:
        pass


def test(word_embed):
    embed_matrix, word_idx = word_embed
    test_data = reasoning_test_data_load('test', word_idx)
    pass

def main():
    run_type = option_parser()  # train/test
    embed_matrix, word_idx = load_word_embedding_table('GLOVE')

    if run_type == 'train':
        train(batch_size=MyConfig.batch_size, epoch=MyConfig.epoch)

    if run_type == 'test':
        test()


if __name__ == '__main__':
    main()

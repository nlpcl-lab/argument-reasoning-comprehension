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
        writer1 = tf.summary.FileWriter(MyConfig.log_dir, sess.graph)
        writer2 = tf.summary.FileWriter(MyConfig.log_dir + '_dev')
        ckpt = tf.train.get_checkpoint_state(MyConfig.train_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("새로운 모델을 생성하는 중 입니다.")
            sess.run(tf.global_variables_initializer())

        for step, batch in enumerate(train_data_gen):
            #  ['warrant0', 'warrant1', 'correctLabelW0orW1', 'reason', 'claim']
            _, cost = model.train(sess, batch)
            if step%20==0: print('Step: {}\nCost: {}'.format(step,cost))



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

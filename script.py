import os
from util import option_parser
from preprocessing import reasoning_batch_generator, reasoning_test_data_load, load_word_embedding_table
from Config import MyConfig
import tensorflow as tf
from model import Model

'''
NeuralNet Script file
'''


def train(batch_size, epoch, word_embed):
    embed_matrix,word_idx = word_embed
    train_data_gen = reasoning_batch_generator(batch_size, epoch, word_idx)
    dev_batch = reasoning_test_data_load('dev', word_idx)
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
            _, cost, acc = model.train(sess, batch[0], batch[1], batch[2], batch[3], batch[4])
            if step%20==0:
                print('Step: {}    Cost: {}'.format(step,cost))
                train_logits, train_acc = model.test(sess, batch[0], batch[1], batch[2], batch[3], batch[4],
                                                     write_logs=True, writer=writer1)
                dev_logits, dev_acc = model.test(sess, dev_batch[0], dev_batch[1], dev_batch[2], dev_batch[3],
                                                 dev_batch[4], write_logs=True, writer=writer2)
                print("train_acc: {}, dev_acc: {}\n".format(train_acc, dev_acc))




def test(word_embed):
    embed_matrix, word_idx = word_embed
    test_data = reasoning_test_data_load('test', word_idx)
    pass


def remove_previous():
    dirname = ['./logdir','./logdir_dev','./house']
    for dir in dirname:
        fnames = os.listdir(dir)
        for f in fnames: os.remove(dir+'/'+f)


def main():
    run_type = option_parser()  # train/test
    word_embed = load_word_embedding_table('GLOVE')

    if run_type == 'train':
        remove_previous()
        train(batch_size=MyConfig.batch_size, epoch=MyConfig.epoch, word_embed=word_embed)

    if run_type == 'test':
        test()


if __name__ == '__main__':
    main()

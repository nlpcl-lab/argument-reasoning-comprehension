import tensorflow as tf
from preprocessing import reasoning_batch_generator, test_data_load, load_word_embedding_table
from esim_model import ESIM
from Config import MyConfig
from util import pretrain_parser
"""
pretrain sentence-embedding model and froze model's parameters
"""


def esim_train(batch_size, epoch, word_embed):
    embed_matrix, word_idx = word_embed
    train_data_gen = reasoning_batch_generator(batch_size, epoch, word_idx)
    dev_batch = test_data_load('dev', word_idx)
    model = ESIM(embed_matrix)

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
            if step % 20 == 0:
                print('Step: {}    Cost: {}'.format(step, cost))
                train_logits, train_acc = model.test(pass,
                                                     write_logs=True, writer=writer1)
                dev_logits, dev_acc = model.test(sess, pass,
                                                 write_logs=True, writer=writer2)
                print("train_acc: {}, dev_acc: {}\n".format(train_acc, dev_acc))


def esim_test(word_embed):
    embed_matrix, word_idx = word_embed
    test_data = test_data_load('test', word_idx)
    pass

def esim_frozen():
    # TODO: Frozen the parameter of ESIM sentence embedding part & Export for comprehension Task
    # reference: https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
    pass


if __name__ == '__main__':
    run_type = pretrain_parser()  # train/test
    word_embed = load_word_embedding_table('GLOVE')

    if run_type == 'esim_train':
        esim_train(batch_size=MyConfig.batch_size, epoch=MyConfig.epoch, word_embed=word_embed)

    if run_type == 'esim_test':
        esim_test(word_embed)

    if run_type == 'esim_frozen_parameter':
        esim_frozen()
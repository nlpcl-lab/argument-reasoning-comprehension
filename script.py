import os
import tensorflow as tf
import argparse
from preprocessing import reasoning_batch_generator, test_data_load, load_word_embedding_table
from Config import MyConfig
from model import Model


parser = argparse.ArgumentParser()

parser.add_argument('--mode',choices=['nli_train','nli_eval','train','eval'],help='Choose the mode to run.', default='nli_train')

parser.add_argument('--reasoning_train_raw_fname',type=str,default='./data/train/train-full.txt')
parser.add_argument('--reasoning_dev_raw_fname',type=str,default='./data/train/dev-full.txt')
parser.add_argument('--reasoning_test_raw_fname',type=str,default='./data/train/test-only-data.txt')

parser.add_argument('--word_embed_glove_fname', type=str, default='./data/emb/glove.6B.300d.txt')
parser.add_argument('--emb_dim', type=int, default=300)
parser.add_argument('--snli_raw_path',type=str,default='./data/nli/snli_1.0/snli_1.0_{}.txt')
parser.add_argument('--snli_bin_path',type=str,default='./data/nli/snli_1.0/snli_1.0_{}.bin')

parser.add_argument("--use_pretrain", type=str, choices=['True', 'False'], default='True')

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_enc_len", type=int, default=50)
parser.add_argument("--learning_rate", type=float, default=5e-4)
parser.add_argument("--max_grad_norm", type=float, default=3)
parser.add_argument("--vocab_size", type=int, default=50000)
parser.add_argument('--l2_coeff', type=float,default=0.1)

parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--decoder_hidden_dim", type=int, default=384)
parser.add_argument("--keep_rate", type=float, default=0.8)
parser.add_argument("--max_epoch", type=int, default=25)

args = parser.parse_args()



def train(batch_size, epoch, word_embed):
    embed_matrix, word_idx = word_embed
    train_data_gen = reasoning_batch_generator(batch_size, epoch, word_idx)
    dev_batch = test_data_load('dev', word_idx)
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
            if step % 20 == 0:
                print('Step: {}    Cost: {}'.format(step, cost))
                train_logits, train_acc = model.test(sess, batch[0], batch[1], batch[2], batch[3], batch[4],
                                                     write_logs=True, writer=writer1)
                dev_logits, dev_acc = model.test(sess, dev_batch[0], dev_batch[1], dev_batch[2], dev_batch[3],
                                                 dev_batch[4], write_logs=True, writer=writer2)
                print("train_acc: {}, dev_acc: {}\n".format(train_acc, dev_acc))

def main():
    args.use_pretrain = True if args.use_pretrain == 'True' else False





if __name__ == '__main__':
    main()

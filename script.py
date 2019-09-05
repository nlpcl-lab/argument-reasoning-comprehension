from time import time
import os
import numpy as np
import tensorflow as tf
import argparse
from model import Model
from esim_model import ESIM
from data_helper import Batcher
import util

parser = argparse.ArgumentParser()

parser.add_argument('--mode',choices=['esim_train', 'esim_eval','train', 'eval'],help='Choose the mode to run.', default='nli_train')
parser.add_argument('--model',choices=['esim','main'],help='Choose the mode to run.')


parser.add_argument('--reasoning_train_raw_fname',type=str,default='./data/main/train-full.txt')
parser.add_argument('--reasoning_dev_raw_fname',type=str,default='./data/main/dev-full.txt')
parser.add_argument('--reasoning_test_raw_fname',type=str,default='./data/main/test-full.txt')
parser.add_argument("--model_path", type=str, default="data/log/{}", help="Path to store the model checkpoints.")
parser.add_argument('--pretrain_ckpt_path', type=str, default='./data/log/esim/exp/train/')
parser.add_argument("--exp_name", type=str, default="exp", help="Path to store the model checkpoints.")
parser.add_argument('--custom_embed_path', type=str, default='./data/emb/my_matrix')

parser.add_argument('--embed_path', type=str, default='./data/emb/glove.6B.300d.txt')
parser.add_argument('--embed_dim', type=int, default=300)
parser.add_argument('--snli_raw_path',type=str,default='./data/nli/snli_1.0/snli_1.0_{}.txt')
parser.add_argument('--snli_bin_path',type=str,default='./data/nli/snli_1.0/snli_1.0_{}.bin')

parser.add_argument("--esim_batch_size", type=int, default=32)
parser.add_argument("--main_batch_size", type=int, default=25)
parser.add_argument("--max_enc_len", type=int, default=80)
parser.add_argument("--learning_rate", type=float, default=5e-4)
parser.add_argument("--max_grad_norm", type=float, default=5)
parser.add_argument("--vocab_size", type=int, default=40000)
parser.add_argument('--l2_coeff', type=float,default=0.0005)

parser.add_argument("--esim_hidden_dim", type=int, default=200)
parser.add_argument("--main_hidden_dim", type=int, default=100)
parser.add_argument("--esim_fcn_hidden_dim", type=int, default=128)
parser.add_argument("--main_fcn_hidden_dim", type=int, default=600)

parser.add_argument("--rnn_keep_rate", type=float, default=0.8)
parser.add_argument("--fcn_keep_rate", type=float, default=0.8)
parser.add_argument("--max_epoch", type=int, default=10)
parser.add_argument("--gpu_nums", type=str, default='0')


args = parser.parse_args()

def train(model, vocab, pretrain_vardicts=None):
    print('train function called.')
    print(model.hps.data_path)
    devpath = model.hps.data_path.replace('train', 'dev')
    assert model.hps.data_path != devpath and os.path.exists(model.hps.data_path) and os.path.exists(devpath)

    train_data_loader = Batcher(vocab, model.hps.data_path, args)
    valid_data_loader = Batcher(vocab, devpath, args)

    print(train_data_loader.example_queue.qsize())
    print(valid_data_loader.example_queue.qsize())
    print(train_data_loader.batch_queue.qsize())
    print(valid_data_loader.batch_queue.qsize())

    with tf.Session(config=util.gpu_config()) as sess:
        train_logdir, dev_logdir = os.path.join(args.model_path, 'logdir/train'), os.path.join(args.model_path, 'logdir/dev')
        train_savedir = os.path.join(args.model_path, 'train/')
        print("[*] Train save directory is: {}".format(train_savedir))
        if not os.path.exists(train_logdir): os.makedirs(train_logdir)
        if not os.path.exists(dev_logdir): os.makedirs(dev_logdir)
        if not os.path.exists(train_savedir): os.makedirs(train_savedir)

        summary_writer1 = tf.summary.FileWriter(train_logdir, sess.graph)
        summary_writer2 = tf.summary.FileWriter(dev_logdir, sess.graph)

        """
        Initialize with pretrain variables
        """
        if 'esim' not in model.hps.mode:
            assign_ops, uninitialized_varlist = util.assign_pretrain_weights(pretrain_vardicts)
            sess.run(assign_ops)
            sess.run(tf.initialize_variables(uninitialized_varlist))
        else:
            sess.run(tf.global_variables_initializer())
        print("Variable initialization end.")
        step = 0
        while True:
            beg_time = time()

            batch = train_data_loader.next_batch()
            sample_per_epoch = 550153 if 'esim' in model.hps.mode else 1211

            res = model.run_step(batch, sess, fcn_keeprate=model.hps.fcn_keep_rate, rnn_keeprate=model.hps.rnn_keep_rate, is_train=True)

            loss, summaries, step = res['loss'], res['summaries'], res['global_step']

            end_time = time()
            print("{} epoch, {} step, {}sec, {} loss".format(int(step * model.hps.batch_size / sample_per_epoch), step,
                                                             round(end_time - beg_time, 3), round(float(loss), 3)))
            summary_writer1.add_summary(summaries, step)

            if step % 5 == 0:
                dev_batch = valid_data_loader.next_batch()
                res = model.run_step(dev_batch, sess, fcn_keeprate=-1, rnn_keeprate=-1, is_train=False)
                loss, summaries, step = res['loss'], res['summaries'], res['global_step']
                assert step % 5 == 0
                print("[VALID] {} loss".format(round(loss, 3)))
                summary_writer2.add_summary(summaries, step)

            if step == 10 or step % 10000 == 0:
                model.saver.save(sess, train_savedir, global_step=step)

            if int(step * model.hps.batch_size / sample_per_epoch) > model.hps.max_epoch:
                model.saver.save(sess, train_savedir, global_step=step)
                print("training end")
                break


def eval(model, vocab):
    datapath = model.hps.data_path.replace('train', 'test')
    data_loader = Batcher(vocab, datapath, args)
    assert 'eval' in model.hps.mode
    assert data_loader.single_pass

    acc_list = []

    with tf.Session(config=util.gpu_config()) as sess:
        util.load_ckpt(model.hps, model.saver, sess)
        print("Running evaluation...\n")
        while True:
            print(len(acc_list), np.mean(acc_list))
            batch = data_loader.next_batch()
            if batch == 'FINISH': break

            res = model.run_eval(batch, sess)
            acc = float(res['accuracy'])
            acc_list.append(acc)
        print("FINAL ACCURACY: {}".format(round(100 * sum(acc_list) / len(acc_list), 2)))


def main():
    if 'train' not in args.mode:
        args.rnn_keep_rate = 1.0
        args.fcn_keep_rate = 1.0
        args.batch_size = 1

    args.data_path = './data/nli/snli_1.0/snli_1.0_train.bin' if 'esim' in args.mode else './data/main/train_binary.bin'

    args.model_path = os.path.join(args.model_path, args.exp_name).format(args.model)
    print(args.model_path)
    if not os.path.exists(args.model_path):
        if 'train' not in args.mode:
            print(args.model_path)
            raise ValueError
        os.makedirs(args.model_path)

    if 'esim' in args.mode:
        args.batch_size = args.esim_batch_size
        assert 'esim' in args.model
    else:
        args.rnn_keep_rate = 1.0
        args.fcn_keep_rate = 1.0
        args.batch_size = args.main_batch_size

    print("Default model path: {}".format(args.model_path))

    print('code start/ {} mode / {} model'.format(args.mode, args.model))
    util.assign_specific_gpu(args.gpu_nums)

    vocab = util.Vocab()

    vardicts = util.get_pretrain_weights(args.pretrain_ckpt_path) if args.mode == 'train' else None

    if args.model == 'main':
        model = Model(vocab, args)
    elif args.model == 'esim':
        model = ESIM(vocab, args)
    else:
        raise ValueError
    print("model build end.")

    if args.mode in ['train', 'esim_train']:
        train(model, vocab, vardicts)
    elif args.mode in ['eval', 'esim_eval']:
        eval(model, vocab)


if __name__ == '__main__':
    main()
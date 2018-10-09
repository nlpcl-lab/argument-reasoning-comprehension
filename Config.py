"""
Project Configuration file
"""


class MyConfig:
    reasoning_train_raw_fname = './data/train/train-full.txt'
    reasoning_dev_raw_txt_fname = './data/dev/dev-full.txt'
    reasoning_test_raw_txt_fname = './data/test/test-only-data.txt'
    tdv_map = {'train': 0, 'dev': 1, 'test': 2}
    reasoning_tdv_fname_list = [reasoning_train_raw_fname, reasoning_dev_raw_txt_fname, reasoning_test_raw_txt_fname]
    word_embed_fasttext_binary_fname = './data/wiki.en/wiki.en.bin'
    word_embed_glove_binary_fname = './data/glove.6B/glove.6B.300d.txt'
    word_embed_vector_len = 300
    log_dir = './logdir'
    train_dir = './house'
    UNK_fname = '_UNK_.txt'
    PAD_fname = '_PAD_.txt'
    n_class = 2

    # Hyperparameters
    l2_coeffi = 0.2
    rnn_train_keeprate = 0.5
    fcn_train_keeprate = 0.5
    epoch = 100
    batch_size = 100
    lr = 0.001
    rnn_layer = 3
    rnn_hidden = 32
    fcn_hidden = rnn_hidden*6

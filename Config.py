
class MyConfig:
    reasoning_train_raw_fname = './data/train/train-full.txt'
    reasoning_dev_raw_txt_fname = './data/dev/dev-full.txt'
    reasoning_test_raw_txt_fname = './data/test/test-only-data.txt'
    tdv_map = {'train': 0, 'dev': 1, 'test': 2}
    reasoning_tdv_fname_list = [reasoning_train_raw_fname,reasoning_dev_raw_txt_fname,reasoning_test_raw_txt_fname]
    word_embed_binary_fname = './data/wiki.en/wiki.en.bin'

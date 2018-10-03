from util import option_parser
from preprocessing import reasoning_batch_generator, reasoning_test_data_load
from Config import MyConfig
'''
NeuralNet Script file
'''


def train(batch_size, epoch):
    train_data = reasoning_batch_generator(batch_size, epoch)
    dev_data = reasoning_test_data_load('dev')
    pass


def test():
    test_data = reasoning_test_data_load('test')
    pass

def main():
    run_type = option_parser()  # train/test
    if run_type == 'train':
        train(batch_size=MyConfig.batch_size, epoch=MyConfig.epoch)

    if run_type == 'test':
        test()


if __name__ == '__main__':
    main()

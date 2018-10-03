from util import option_parser
from preprocessing import reasoning_batch_generator
from Config import MyConfig
'''
NeuralNet Script file
'''


def train(batch_size, epoch):
    train_input = reasoning_batch_generator('train', batch_size, epoch)
    dev_input = reasoning_batch_generator('dev')
    pass


def test():
    pass


def main():
    run_type = option_parser()  # train/test
    if run_type == 'train':
        train(batch_size=MyConfig.batch_size, epoch=MyConfig.epoch)

    if run_type == 'test':
        test()


if __name__ == '__main__':
    main()

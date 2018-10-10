import argparse
'''
ETC util file
'''


def option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='train or test')
    args = parser.parse_args()
    runtype = args.mode
    print(runtype)
    assert runtype in ['train', 'test']
    return runtype

def pretrain_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='train or test')
    args = parser.parse_args()
    runtype = args.mode
    print(runtype)
    assert runtype in ['train', 'test','frozen']
    return runtype

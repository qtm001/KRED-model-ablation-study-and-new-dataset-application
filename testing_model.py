import argparse
from train_test import *
from utils.util import *
import pickle
from parse_config import ConfigParser

import os

def main(config):
    #data = load_data_mind(config)
    task = config['trainer']['task']
    with open(f"./data/data_mind_small_{task}.pkl", 'rb') as f:
        data = pickle.load(f)
    # test_data = data[-1]
    print('data successfully loaded; start testing')
    testing(data[-1], config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KRED')

    parser.add_argument('-c', '--config', default="./config.json", type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(parser)
    main(config)
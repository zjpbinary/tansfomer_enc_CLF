import argparse

parser = argparse.ArgumentParser(description="神经网络参数配置")
parser.add_argument("--gpu", default='cpu')
args = parser.parse_args()

class TextCNNConfig(object):
    trainfile = 'data/train'
    testfile = 'data/train'
    d_model = 256
    dropout = 0.5
    lr = 0.000001
    batch_size = 50
    epoch = 100
    device = args.gpu

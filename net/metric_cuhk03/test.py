#!/usr/bin/python
from experiment import test_on_VIPeR, deep_method
import argparse, os, sys

_TFP = os.path.dirname(__file__)

def parse_args():
    parser = argparse.ArgumentParser(description='Test on Net')
    parser.add_argument(
        '-N', '--net', action='store', dest='net',
        type=str, default=_TFP+'/deploy.prototxt',
        help='The sepecified structure (.prototxt)')
    parser.add_argument(
        '-M', '--model', action='store', dest='model',
        type=str, default=_TFP+'/snapshots/train_val_iter_2000.caffemodel',
        help='The trained model file (.caffemodel)'
    )

    return parser.parse_args()

def check_arg(args):
    print '[NET]   :',args.net
    print '[MODEL] :',args.model
    assert os.path.isfile(args.net)
    assert os.path.isfile(args.model)

if __name__ == '__main__':
    args = parse_args()
    check_arg(args)

    method = deep_method(args.net, args.model)

    result=test_on_VIPeR(method)
    print result.shape, result[:50]
    pass

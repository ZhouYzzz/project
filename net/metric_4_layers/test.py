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
        type=str, default=_TFP+'/snapshots/train_val_iter_20000.caffemodel',
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

    print '===================='
    print ' CMC Curve on VIPeR '
    print '===================='
    for i in [1, 2, 10, 20, 30, 50, 100, 200]:
        print 'TOP%4d'%i,'%10.2f'%(result[i-1]*100),'%'
        print '--------------------'

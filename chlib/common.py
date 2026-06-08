import os
import time
import numpy as np
import glob
import argparse
from argparse import ArgumentParser
from collections import defaultdict

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val != np.nan and val != np.inf:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

def init_stats():
    stats = defaultdict(AverageMeter)
    ctx = argparse.Namespace()
    return stats, ctx

def update_stats(stats, ctx):
    for m in ctx.__dict__:
        stats[m].update(ctx.__dict__[m])

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def join_path(*dirs):
    if len(dirs) == 0:
        return ''
    path = dirs[0]
    for d in dirs[1:]:
        path = os.path.join(path, d)
    return path

def make_filepath(fpath, dir_name=None, ext_name=None, tag=None):
    if dir_name is None:
        dir_name = os.path.dirname(fpath)
        if dir_name == '':
            dir_name = '.'
    fname = os.path.basename(fpath)
    base, ext = os.path.splitext(fname)
    if ext_name is None:
        ext_name = ext
    elif ext_name != '' and ext_name[0] != '.':
        ext_name = '.' + ext_name
    name = base
    if tag == '':
        name = name.split('-')[0]
    elif tag is not None:
        name = '%s-%s' % (name, tag)
    if ext_name != '':
        name = '%s%s' % (name, ext_name)
    return join_path(dir_name, name)

def read_image_list(fn_list):
    with open(fn_list, 'r') as f:
        res = [l.strip() for l in f.readlines()]
    return res

def plug_image_path(parser: ArgumentParser, op_name='--im_path', op_name_s='-p'):
    parser.add_argument(op_name, op_name_s, dest='im_path', nargs='+', type=str, default=None, help='image path')

def plug_image_list(parser: ArgumentParser, op_name='--im_list', op_name_s='-l'):
    parser.add_argument(op_name, op_name_s, dest='im_list', type=str, default=None, help='image list file')

def plug_image_dir(parser: ArgumentParser, op_name='--im_dir', op_name_s='-d'):
    parser.add_argument(op_name, op_name_s, dest='im_dir', type=str, default=None, help='image dir')

def prepare_image_args(args):
    if args.im_list is not None:
        args.images = read_image_list(args.im_list)
    elif args.im_dir is not None:
        args.images = []
        for im_name in glob.glob(join_path(args.im_dir, f'*')):
            if im_name.lower().endswith('.png') or im_name.lower().endswith('.jpg') or im_name.lower().endswith('.jpeg'):
                args.images.append(os.path.abspath(im_name)) 
    elif args.im_path is not None:
        args.images = [os.path.abspath(im_name) for im_name in args.im_path]
    return args

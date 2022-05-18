from glob import glob
import numpy as np
import os
from collections import defaultdict


def process_det(det_file):
    re = defaultdict(list)
    with open(det_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().split(',')
            re[int(l[0])].append([int(x) for x in l[2:6]])
    np.save(det_file.replace('.txt', '.npy'), re)


def process_gt(gt_file):
    re = defaultdict(dict)
    with open(gt_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().split(',')
            re[int(l[0])][int(l[1])] = [int(x) for x in l[2:6]]
    np.save(gt_file.replace('.txt', '.npy'), re)


def main():
    train_folders = glob(os.path.join('../data/tracking/*', '*'))

    for folder in train_folders:
        if os.path.exists('{}/det'.format(folder)):
            process_det('{}/det/det.txt'.format(folder))
        if os.path.exists('{}/gt'.format(folder)):
            process_gt('{}/gt/gt.txt'.format(folder))


if __name__ == '__main__':
    main()

#!/usr/bin/env python

# from __future__ import print_function

import os
import numpy as np
import io_tool as it

from sklearn import preprocessing as pp


def fit(feat, scaler):
    scaler.partial_fit(feat)


if __name__ == '__main__':
    # Load the dataset
    print("Loading data...")
    feat_type = 'image.16000_512.16_frames_per_seg.h256_w256'

    time_range = (30, 60)
    num_files = 2000

    # Base dirs
    base_dir = '/home/ciaua/NAS/home/data/youtube8m/'
    out_dir = os.path.join(base_dir, 'image_scaler')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    feat_dir = os.path.join(
        base_dir,
        'feature', 'video.time_{}_to_{}'.format(*time_range),
        feat_type
    )

    fold_dir = os.path.join(base_dir, 'fold.tr16804_va2100_te2100')

    fold_fp = os.path.join(fold_dir, 'fold.tr.txt')

    fn_list = it.read_lines(fold_fp)

    # scaler output
    scaler_fp = os.path.join(out_dir, 'scaler.RGB.pkl')

    mean_fp = os.path.join(out_dir, 'mean.RGB.csv')
    std_fp = os.path.join(out_dir, 'std.RGB.csv')

    # Make scaler
    scaler = pp.StandardScaler()
    np.random.shuffle(fn_list)

    # Fit
    print('Fit...')
    for ii, fn in enumerate(fn_list[:num_files]):
        in_fp = os.path.join(feat_dir, '{}.npy'.format(fn))
        try:
            feat = np.load(in_fp)
            feat = feat.transpose((0, 2, 3, 1))

            k = feat.shape[-1]
            feat = feat.reshape((-1, k))
            # np.random.shuffle(feat)
            # raw_input(123)
            fit(feat, scaler)
        except:
            continue
        print(ii)

    it.pickle(scaler_fp, scaler)
    it.write_csv(mean_fp, [scaler.mean_.tolist()])
    it.write_csv(std_fp, [scaler.scale_.tolist()])

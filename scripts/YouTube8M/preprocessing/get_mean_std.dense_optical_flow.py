#!/usr/bin/env python

# from __future__ import print_function

import os
import numpy as np
import io_tool as it
from jjtorch import utils

from sklearn import preprocessing as pp


def fit(feat, scaler):
    scaler.partial_fit(feat)


if __name__ == '__main__':
    # Load the dataset
    print("Loading data...")
    # feat_type = 'dense_optical_flow.16000_512.fill_2.16_frames_per_seg.5_flows_per_frame.h_w_max_256.plus127.10x10xD2xD3_to_100xD2*D3_png'
    feat_type = 'dense_optical_flow.16000_512.fill_16.16_frames_per_seg.5_flows_per_frame.h_w_max_256.plus128.10x10xD2xD3_to_100xD2*D3_png'

    time_range = (0, 60)
    # num_files = 2000

    # Base dirs
    base_dir = '/home/ciaua/NAS/home/data/youtube8m/'
    out_dir = os.path.join(base_dir, 'scaler', feat_type)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    feat_dir = os.path.join(
        base_dir,
        'feature.5s_fragment', 'video.time_{}_to_{}'.format(*time_range),
        feat_type
    )

    fold_dir = os.path.join(base_dir, 'fold.tr16804_va2100_te2100')

    fold_fp = os.path.join(fold_dir, 'fold.tr.txt')

    fn_list = it.read_lines(fold_fp)

    # scaler output
    scaler_fp = os.path.join(out_dir, 'scaler.pkl')

    mean_fp = os.path.join(out_dir, 'mean.csv')
    std_fp = os.path.join(out_dir, 'std.csv')

    # Make scaler
    scaler = pp.StandardScaler()
    np.random.shuffle(fn_list)

    # Fit
    print('Fit...')
    # for ii, fn in enumerate(fn_list[:num_files]):
    for ii, fn in enumerate(fn_list):
        in_fp = os.path.join(feat_dir, fn, '30_35.png')
        # raw_input(123)
        try:
            feat = utils._load_one_plus128_npy(in_fp)
            raw_input(123)
            feat = feat.transpose((0, 2, 3, 1))

            k = feat.shape[-1]
            feat = feat.reshape((-1, k))
            # np.random.shuffle(feat)
            # raw_input(123)
            fit(feat, scaler)
        except Exception as e:
            print(repr(e))
            continue
        print(ii)

    it.pickle(scaler_fp, scaler)
    it.write_csv(mean_fp, [scaler.mean_.tolist()])
    it.write_csv(std_fp, [scaler.scale_.tolist()])

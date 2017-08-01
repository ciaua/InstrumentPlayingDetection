#!/usr/bin/env python

import os
from jjtorch import load_data as ld


if __name__ == '__main__':
    base_dir = "/home/ciaua/NAS/home/data/youtube8m/"

    feat_type_list = [
        "logmelspec10000.16000_512_512_128.0.raw",
        'logmelspec10000.16000_2048_512_128.0.raw',
        'logmelspec10000.16000_8192_512_128.0.raw'
    ]
    # feat_type = 'logmelspec10000.16000_512_512_128.0.raw'
    # feat_type = 'logmelspec10000.16000_2048_512_128.0.raw'
    # feat_type = 'logmelspec10000.16000_8192_512_128.0.raw'
    # feat_type = 'cqt.16000_512_A0_24_176.0.raw'
    time_range = (30, 60)

    # Dirs and fps
    data_dir = os.path.join(
        base_dir, 'exp_data_common.audio.time_{}_to_{}'.format(*time_range))

    # Loading data
    print("Loading data...")
    prefix = 'jy.yt8m.audio'
    if not ld.all_exist(feat_type_list, prefix=prefix):
        ld.load2memory_tr_va(data_dir, feat_type_list, prefix=prefix)

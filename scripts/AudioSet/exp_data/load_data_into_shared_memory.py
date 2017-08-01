#!/usr/bin/env python

import os
from jjtorch import load_data as ld


if __name__ == '__main__':
    base_dir = "/home/ciaua/NAS/home/data/AudioSet/"

    feat_type_list = [
        "logmelspec10000.16000_512_512_128.0.raw",
        'logmelspec10000.16000_2048_512_128.0.raw',
        'logmelspec10000.16000_8192_512_128.0.raw'
    ]
    # feat_type = 'logmelspec10000.16000_512_512_128.0.raw'
    # feat_type = 'logmelspec10000.16000_2048_512_128.0.raw'
    # feat_type = 'logmelspec10000.16000_8192_512_128.0.raw'
    # feat_type = 'cqt.16000_512_A0_24_176.0.raw'
    clip_limit = 5000

    # Dirs and fps
    data_dir = os.path.join(
        base_dir, 'exp_data_common.audio.target_time.{}'.format(clip_limit))

    # Loading data
    print("Loading data...")
    prefix = 'jy.audioset'
    if not ld.all_exist(feat_type_list, prefix=prefix):
        ld.load2memory_tr_va(data_dir, feat_type_list, prefix=prefix)

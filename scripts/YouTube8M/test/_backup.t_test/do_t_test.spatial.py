#!/usr/bin/env python

import os
import io_tool as it
import numpy as np

from scipy.stats import ttest_rel

if __name__ == '__main__':
    fragment_unit = 5  # second

    # Settings

    # model_id_i = '20170319_085641'  # image

    # model_id_o = '20170403_183405'  # videoanno
    # model_id_o = '20170403_183041'  # audioanno

    model_id_1 = '20170319_085641__20170403_183041'  # audioanno

    # model_id_2 = '20170319_085641__20170403_183405'  # videoanno
    # model_id_2 = '20170319_085641'  # image
    model_id_2 = '20170403_183041'  # audioanno

    param_type = 'best_measure'

    # Dirs and fps
    base_dir = "/home/ciaua/NAS/home/data/youtube8m/"

    # Tag
    tag_fp = os.path.join(base_dir, 'tag_list.instrument.csv')
    tag_list = [term[0] for term in it.read_csv(tag_fp)]

    # Output
    out_dir_1 = os.path.join(
        base_dir,
        'test_result.distance.spatial.fragment_{}s'.format(fragment_unit),
        param_type,
        model_id_1
    )

    out_dir_2 = os.path.join(
        base_dir,
        'test_result.distance.spatial.fragment_{}s'.format(fragment_unit),
        param_type,
        model_id_2
    )

    # Do t test
    for jj, tag in enumerate(tag_list):
        in_fp_1 = os.path.join(out_dir_1, '{}.csv'.format(tag))
        in_fp_2 = os.path.join(out_dir_2, '{}.csv'.format(tag))

        dist_1 = [float(term[-1]) for term in it.read_csv(in_fp_1)]
        dist_2 = [float(term[-1]) for term in it.read_csv(in_fp_2)]

        _, p = ttest_rel(dist_1, dist_2)

        print(tag)
        print('{}: {}. {}: {}. p: {}'.format(
            model_id_1, np.mean(dist_1), model_id_2, np.mean(dist_2),
            p/2
        ))

#!/usr/bin/env python

import os
import io_tool as it
import numpy as np

from scipy.stats import ttest_rel

if __name__ == '__main__':
    measure_type = 'auc_classwise'
    # measure_type = 'ap_classwise'

    # Settings
    model_id_1 = '20170708_154229'
    # model_id_1 = '20170713_002039'
    # model_id_2 = '20170708_154229'
    model_id_2 = '20170714_144445'

    param_type = 'best_measure'

    # Dirs and fps
    base_dir = "/home/ciaua/NAS/home/data/TMM17_instrument_playing/YT8M/"
    base_data_dir = "/home/ciaua/NAS/home/data/youtube8m/"

    # Tag
    tag_fp = os.path.join(base_data_dir, 'tag_list.instrument.csv')
    tag_list = [term[0] for term in it.read_csv(tag_fp)]

    # Output
    base_in_dir_1 = os.path.join(
        base_dir,
        'test_result.video_wise.temporal.no_resize',
        param_type,
        model_id_1)

    base_in_dir_2 = os.path.join(
        base_dir,
        'test_result.video_wise.temporal.no_resize',
        param_type,
        model_id_2)

    # Process groundtruth
    in_dir_1 = os.path.join(base_in_dir_1, measure_type)
    in_dir_2 = os.path.join(base_in_dir_2, measure_type)
    for ii, tag in enumerate(tag_list):
        in_fp_1 = os.path.join(in_dir_1, '{}.csv'.format(tag))
        in_fp_2 = os.path.join(in_dir_2, '{}.csv'.format(tag))

        score_1 = [float(term[1])
                   for term in it.read_csv(in_fp_1) if term[1] != 'nan']
        score_2 = [float(term[1])
                   for term in it.read_csv(in_fp_2) if term[1] != 'nan']
        _, p = ttest_rel(score_1, score_2)

        print(tag)
        print('{}: {}. {}: {}. p: {}'.format(
            model_id_1, np.mean(score_1), model_id_2, np.mean(score_2),
            p/2
        ))

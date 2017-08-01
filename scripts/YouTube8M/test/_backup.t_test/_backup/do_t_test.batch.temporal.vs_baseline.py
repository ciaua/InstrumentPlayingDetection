#!/usr/bin/env python

import os
import io_tool as it
import numpy as np

from scipy.stats import ttest_rel

if __name__ == '__main__':
    alpha = 0.05
    # alpha = 0.1
    fragment_unit = 5  # second

    # measure_type_list = ['auc_classwise', 'ap_classwise']
    measure_type_list = ['auc_classwise']

    # Settings
    model_id_list = ['20170319_085641',  # object
                     '20170403_183405',  # videoanno
                     '20170403_183041',  # audioanno
                     '20170319_085641__20170403_183405',
                     '20170319_085641__20170403_183041']

    # baseline model id
    baseline_model_id = '20170319_085641'

    #
    param_type = 'best_measure'

    # Dirs and fps
    base_dir = "/home/ciaua/NAS/home/data/youtube8m/"

    # Tag
    tag_fp = os.path.join(base_dir, 'tag_list.instrument.csv')
    tag_list = [term[0] for term in it.read_csv(tag_fp)]

    # Do t-test
    out_list = list()
    for ii, tag in enumerate(tag_list):
        for measure_type in measure_type_list:

            # do t-test
            in_dir = os.path.join(
                base_dir,
                'test_result.video_wise.temporal.fragment_{}s'.format(
                    fragment_unit), param_type, baseline_model_id,
                measure_type)
            in_fp = os.path.join(in_dir, '{}.csv'.format(tag))
            baseline_score = [
                float(term[1])
                for term in it.read_csv(in_fp) if term[1] != 'nan']

            is_better_list = list()
            print(tag)
            for model_id in model_id_list:
                in_dir = os.path.join(
                    base_dir,
                    'test_result.video_wise.temporal.fragment_{}s'.format(
                        fragment_unit), param_type, model_id,
                    measure_type)
                in_fp = os.path.join(in_dir, '{}.csv'.format(tag))
                score = [float(term[1])
                         for term in it.read_csv(in_fp) if term[1] != 'nan']

                t, p = ttest_rel(baseline_score, score)

                # One-tailed
                p = p/2

                if p < alpha and t < 0:
                    is_better_list.append(1)
                else:
                    is_better_list.append(0)
                print(model_id, t, p)
            # raw_input(123)
            out_list.append(is_better_list)
    out = np.array(out_list).T

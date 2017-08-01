#!/usr/bin/env python

import os
import io_tool as it
import numpy as np

from scipy.stats import ttest_rel

if __name__ == '__main__':
    alpha = 0.05
    # alpha = 0.1
    fragment_unit = 5  # second

    measure_type_list = ['auc_classwise', 'ap_classwise']

    # Settings
    model_id_list = ['20170319_085641',  # object
                     '20170403_183405',  # videoanno
                     '20170403_183041',  # audioanno
                     '20170319_085641__20170403_183405',
                     '20170319_085641__20170403_183041']

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

            # find the best model id
            mean_scores = list()
            for model_id in model_id_list:
                in_dir = os.path.join(
                    base_dir,
                    'test_result.video_wise.temporal.fragment_{}s'.format(
                        fragment_unit), param_type, model_id,
                    measure_type)
                in_fp = os.path.join(in_dir, '{}.csv'.format(tag))
                score = [float(term[1])
                         for term in it.read_csv(in_fp) if term[1] != 'nan']

                mean_scores.append(np.mean(score))
            best_idx = np.argmax(mean_scores)
            best_model_id = model_id_list[best_idx]

            # do t-test
            in_dir = os.path.join(
                base_dir,
                'test_result.video_wise.temporal.fragment_{}s'.format(
                    fragment_unit), param_type, best_model_id,
                measure_type)
            in_fp = os.path.join(in_dir, '{}.csv'.format(tag))
            best_score = [float(term[1])
                          for term in it.read_csv(in_fp) if term[1] != 'nan']

            is_best_list = list()
            for model_id in model_id_list:
                in_dir = os.path.join(
                    base_dir,
                    'test_result.video_wise.temporal.fragment_{}s'.format(
                        fragment_unit), param_type, model_id,
                    measure_type)
                in_fp = os.path.join(in_dir, '{}.csv'.format(tag))
                score = [float(term[1])
                         for term in it.read_csv(in_fp) if term[1] != 'nan']

                _, p = ttest_rel(best_score, score)

                # One-tailed
                p = p/2

                if model_id == best_model_id:
                    is_best_list.append(1)
                else:
                    if p > alpha:
                        is_best_list.append(1)
                    else:
                        is_best_list.append(0)
            out_list.append(is_best_list)
    out = np.array(out_list).T

'''
                score_1 = [float(term[1])
                        for term in it.read_csv(in_fp_1) if term[1] != 'nan']
                score_2 = [float(term[1])
                        for term in it.read_csv(in_fp_2) if term[1] != 'nan']
                _, p = ttest_rel(score_1, score_2)

                # One-tailed
                p = p/2

                print(tag)
                print('{}: {}. {}: {}. p: {}'.format(
                    model_id_1, np.mean(score_1), model_id_2, np.mean(score_2),
                    p
                ))
                '''

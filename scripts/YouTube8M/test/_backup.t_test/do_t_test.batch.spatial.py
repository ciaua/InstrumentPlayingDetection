#!/usr/bin/env python

import os
import io_tool as it
import numpy as np

from scipy.stats import ttest_rel

if __name__ == '__main__':
    alpha = 0.05

    fragment_unit = 5  # second

    # Settings
    model_id_list = [
        'center',
        '20170319_085641',  # object
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

        # find the best model id
        mean_scores = list()
        for model_id in model_id_list:
            in_dir = os.path.join(
                base_dir,
                'test_result.distance.spatial.fragment_{}s'.format(
                    fragment_unit),
                param_type,
                model_id
            )
            in_fp = os.path.join(in_dir, '{}.csv'.format(tag))
            dist = [float(term[-1]) for term in it.read_csv(in_fp)]

            mean_scores.append(np.mean(dist))
        best_idx = np.argmin(mean_scores)
        best_model_id = model_id_list[best_idx]

        # do t-test
        in_dir = os.path.join(
            base_dir,
            'test_result.distance.spatial.fragment_{}s'.format(
                fragment_unit),
            param_type,
            best_model_id
        )
        in_fp = os.path.join(in_dir, '{}.csv'.format(tag))
        best_dist = [float(term[-1]) for term in it.read_csv(in_fp)]

        is_best_list = list()
        for model_id in model_id_list:
            in_dir = os.path.join(
                base_dir,
                'test_result.distance.spatial.fragment_{}s'.format(
                    fragment_unit),
                param_type,
                model_id
            )
            in_fp = os.path.join(in_dir, '{}.csv'.format(tag))
            dist = [float(term[-1]) for term in it.read_csv(in_fp)]

            _, p = ttest_rel(best_dist, dist)

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

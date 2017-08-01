#!/usr/bin/env python

import os
import io_tool as it
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score

if __name__ == '__main__':
    time_range = (0, 60)
    fragment_unit = 5  # second
    num_fragments = (time_range[1]-time_range[0]) // fragment_unit

    measure_type_list = ['auc_classwise', 'ap_classwise']

    test_fn_list = ['0_5', '5_10', '30_35', '35_40']

    phase = 'te'

    # Settings
    sr = 16000
    hop = 512
    num_tags = 9

    visual_total_stride = 32

    num_frames_per_seg = 16
    target_size = None  # (height, width)

    model_id = '20170319_085641'

    param_type = 'best_measure'
    # param_type = 'best_loss'

    # Dirs and fps
    base_dir = "/home/ciaua/NAS/home/data/youtube8m/"
    id_dir = os.path.join(base_dir, 'picked_id')
    id_dict_fp = os.path.join(id_dir, 'picked_id.{}.json'.format(phase))
    id_dict = it.read_json(id_dict_fp)

    # Tag
    tag_fp = os.path.join(base_dir, 'tag_list.instrument.csv')
    tag_list = [term[0] for term in it.read_csv(tag_fp)]

    all_id_list = list()
    for tag in tag_list:
        all_id_list += id_dict[tag]

    # Groundtruth and prediction
    base_gt_dir = os.path.join(
        base_dir,
        'manual_annotation.action_location.h_w.down_right',
        phase)

    base_pred_dir = os.path.join(
        base_dir,
        'prediction_without_resize',
        'rgb_image.{}_{}.fragment_{}s'.format(
            time_range[0], time_range[1], fragment_unit),
        model_id, param_type)

    # Output
    out_dir = os.path.join(
        base_dir,
        'test_result.temporal.fragment_{}s'.format(fragment_unit),
        model_id, param_type)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Process prediction
    print('Process prediction...')
    pred_list = list()
    for uu, song_id in enumerate(all_id_list):
        print(uu, song_id)
        pred_dir = os.path.join(base_pred_dir, song_id)

        for fn in test_fn_list:
            #
            pred_fp = os.path.join(pred_dir, '{}.npy'.format(fn))
            pred = np.load(pred_fp)

            num_segments = pred.shape[0]

            pred = pred.max(axis=2).max(axis=2)
            pred_list.append(pred)
    pred_all = np.concatenate(pred_list, axis=0)

    # Process groundtruth
    print('Process groundtruth...')
    gt_list = list()
    for jj, tag in enumerate(tag_list):
        id_list = id_dict[tag]

        for uu, song_id in enumerate(id_list):
            print(uu, song_id)
            gt_dir = os.path.join(base_gt_dir, tag, song_id)

            for fn in test_fn_list:

                for ii in range(num_segments):
                    gt_fp = os.path.join(gt_dir, '{}.{}.npy'.format(fn, ii))
                    gt = np.load(gt_fp)

                    vec = np.zeros((1, num_tags), dtype=int)
                    vec[0, jj] = (gt.size > 0)
                    gt_list.append(vec)
    gt_all = np.concatenate(gt_list, axis=0)

    for measure_type in measure_type_list:
        if measure_type == 'auc_classwise':
            measure = roc_auc_score(gt_all, pred_all, None)
        elif measure_type == 'ap_classwise':
            measure = average_precision_score(gt_all, pred_all, None)
        out_fp = os.path.join(out_dir, '{}.csv'.format(measure_type))
        it.write_csv(out_fp, zip(tag_list, measure))

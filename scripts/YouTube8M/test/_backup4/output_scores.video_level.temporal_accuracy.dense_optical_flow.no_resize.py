#!/usr/bin/env python

import os
import io_tool as it
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score

if __name__ == '__main__':
    time_range = (0, 60)
    fragment_unit = 5  # second

    measure_type_list = ['auc_classwise', 'ap_classwise']

    test_fn_list = ['0_5', '5_10', '30_35', '35_40']

    phase = 'te'

    # Settings
    num_tags = 9

    # model_id_fill = ('20170709_161154', 1)
    # model_id_fill = ('20170708_073900', 2)
    # model_id_fill = ('20170708_154229', 4)
    # model_id_fill = ('20170711_080438', 8)
    # model_id_fill = ('20170711_175508', 16)

    # model_id_fill = ('20170713_153618', 4)  # 0.1 threshold
    # model_id_fill = ('20170713_002039', 4)  # 0.5 threshold
    # model_id_fill = ('20170713_055026', 4)  # 0.9 threshold

    # model_id_fill = ('20170713_224322', 4)  # 0.5 threshold with instrument mask

    model_id_fill = ('20170714_144445', 4)  # instrument*audio as target
    # model_id_fill = ('20170714_155252', 4)  # instrument*audio as target v2

    param_type = 'best_measure'

    # Dirs and fps
    base_dir = "/home/ciaua/NAS/home/data/TMM17_instrument_playing/YT8M/"
    base_data_dir = "/home/ciaua/NAS/home/data/youtube8m/"
    id_dir = os.path.join(base_data_dir, 'picked_id')
    id_dict_fp = os.path.join(id_dir, 'picked_id.{}.json'.format(phase))
    id_dict = it.read_json(id_dict_fp)

    model_id_o, fill_factor = model_id_fill

    # Tag
    tag_fp = os.path.join(base_data_dir, 'tag_list.instrument.csv')
    tag_list = [term[0] for term in it.read_csv(tag_fp)]

    # tag_list = tag_list[0:1]#

    all_id_list = list()
    for tag in tag_list:
        all_id_list += id_dict[tag]

    # Groundtruth and prediction
    base_gt_dir = os.path.join(
        base_data_dir,
        'manual_annotation.action_location.h_w.down_right',
        phase)

    base_pred_dir_o = os.path.join(
        base_dir,
        'predictions.action.no_resize',
        'dense_optical_flow.fill_{}.{}_{}.fragment_{}s'.format(
            fill_factor, time_range[0], time_range[1], fragment_unit),
        model_id_o, param_type)

    # Output
    base_out_dir = os.path.join(
        base_dir,
        'test_result.video_wise.temporal.fill_{}.fragment_{}s.no_resize'.format(
            fill_factor, fragment_unit),
        param_type,
        '{}'.format(model_id_o))

    # Process prediction
    print('Process prediction...')
    pred_dict = dict()
    for uu, song_id in enumerate(all_id_list):
        print(uu, song_id)
        pred_dir_o = os.path.join(base_pred_dir_o, song_id)

        pred_list = list()
        for fn in test_fn_list:
            pred_fp_o = os.path.join(pred_dir_o, '{}.npy'.format(fn))
            pred_o = np.load(pred_fp_o)
            # print(fn)

            num_segments = pred_o.shape[0]

            pred = pred_o.max(axis=2).max(axis=2)
            pred_list.append(pred)
        pred_all = np.concatenate(pred_list, axis=0)
        pred_dict[song_id] = pred_all

    # Process groundtruth
    print('Process groundtruth...')
    gt_dict = dict()
    for jj, tag in enumerate(tag_list):
        id_list = id_dict[tag]

        gt_dict[tag] = dict()

        for uu, song_id in enumerate(id_list):
            print(uu, song_id)
            gt_dir = os.path.join(base_gt_dir, tag, song_id)

            gt_list = list()
            for fn in test_fn_list:
                # print(fn)

                for ii in range(num_segments):
                    gt_fp = os.path.join(gt_dir, '{}.{}.npy'.format(fn, ii))
                    gt = np.load(gt_fp)

                    vec = np.zeros((1, num_tags), dtype=int)
                    vec[0, jj] = (gt.size > 0)
                    gt_list.append(vec)
            gt_all = np.concatenate(gt_list, axis=0)
            gt_dict[tag][song_id] = gt_all

    for measure_type in measure_type_list:
        out_dir = os.path.join(base_out_dir, measure_type)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for ii, tag in enumerate(tag_list):
            print(measure_type, tag)
            id_list = gt_dict[tag]

            out_fp = os.path.join(out_dir, '{}.csv'.format(tag))

            out_list = list()
            for id_ in id_list:
                g = gt_dict[tag][id_][:, ii:ii+1]
                p = pred_dict[id_][:, ii:ii+1]
                try:
                    if measure_type == 'auc_classwise':
                        measure = roc_auc_score(g, p, None)
                    elif measure_type == 'ap_classwise':
                        measure = average_precision_score(g, p, None)
                    if measure != np.nan:
                        out_list.append((id_, measure))

                    print(id_, measure)
                except Exception as e:
                    continue
            print(tag)
            it.write_csv(out_fp, out_list)

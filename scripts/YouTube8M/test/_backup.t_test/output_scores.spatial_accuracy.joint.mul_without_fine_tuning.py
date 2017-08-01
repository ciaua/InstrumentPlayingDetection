#!/usr/bin/env python

import os
import io_tool as it
import numpy as np
from scipy.misc import imresize

# from sklearn.metrics import roc_auc_score, average_precision_score


def compute_euc_distances(arr, target_vec):
    temp = arr-target_vec
    dist_list = np.sqrt(np.sum(temp**2, axis=1))
    return dist_list


def get_prediction(song_id, sub_time_range_str,
                   base_pred_dir_i, base_pred_dir_o):
    pred_dir_i = os.path.join(base_pred_dir_i, song_id)
    pred_dir_o = os.path.join(base_pred_dir_o, song_id)

    pred_fp_i = os.path.join(pred_dir_i,
                             '{}.npy'.format(sub_time_range_str))
    pred_fp_o = os.path.join(pred_dir_o,
                             '{}.npy'.format(sub_time_range_str))
    pred_i = np.load(pred_fp_i)
    pred_o = np.load(pred_fp_o)

    pred = (pred_i*pred_o)
    return pred


if __name__ == '__main__':
    time_range = (0, 60)
    fragment_unit = 5  # second
    num_fragments = (time_range[1]-time_range[0]) // fragment_unit

    measure_type = 'euclidean_dist'

    test_fn_list = ['0_5', '5_10', '30_35', '35_40']

    phase = 'te'

    # Settings
    sr = 16000
    hop = 512
    num_tags = 9

    visual_total_stride = 32

    num_frames_per_seg = 16
    # target_size = None  # (height, width)

    model_id_i = '20170319_085641'  # image

    # model_id_o = '20170403_183405'
    model_id_o = '20170403_183041'

    param_type = 'best_measure'
    # param_type = 'best_loss'

    # Dirs and fps
    base_dir = "/home/ciaua/NAS/home/data/youtube8m/"
    id_dir = os.path.join(base_dir, 'picked_id')
    id_dict_fp = os.path.join(id_dir, 'picked_id.{}.json'.format(phase))
    id_dict = it.read_json(id_dict_fp)

    base_image_dir = os.path.join(
        base_dir,
        'feature.5s_fragment',
        'video.time_0_to_60',
        'image.16000_512.16_frames_per_seg.h_w_max_256')

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

    base_pred_dir_i = os.path.join(
        base_dir,
        'predictions_without_resize', 'rgb_image.{}_{}.fragment_{}s'.format(
            time_range[0], time_range[1], fragment_unit),
        model_id_i, param_type)
    base_pred_dir_o = os.path.join(
        base_dir,
        'predictions_without_resize',
        'dense_optical_flow.{}_{}.fragment_{}s'.format(
            time_range[0], time_range[1], fragment_unit),
        model_id_o, param_type)

    # Output
    out_dir = os.path.join(
        base_dir,
        'test_result.distance.spatial.fragment_{}s'.format(fragment_unit),
        param_type,
        '{}__{}'.format(model_id_i, model_id_o),
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Process groundtruth
    print('Process groundtruth...')
    measure_list = list()
    classwise_measure_list = list()
    for jj, tag in enumerate(tag_list):
        tag_idx = tag_list.index(tag)
        id_list = id_dict[tag]

        out_fp = os.path.join(out_dir, '{}.csv'.format(tag))

        in_class_measure_list = list()
        for uu, song_id in enumerate(id_list):
            print(uu, song_id)
            gt_dir = os.path.join(base_gt_dir, tag, song_id)

            image_dir = os.path.join(base_image_dir, song_id)

            for fn in test_fn_list:
                fn_raw = os.path.splitext(fn)[0]
                pred = get_prediction(
                    song_id, fn, base_pred_dir_i, base_pred_dir_o)
                num_segments = pred.shape[0]

                image_fp = os.path.join(image_dir, '{}.npy'.format(fn))
                images = np.load(image_fp)
                for ii in range(num_segments):
                    gt_fp = os.path.join(gt_dir, '{}.{}.npy'.format(fn, ii))
                    one_gt = np.load(gt_fp)

                    if one_gt.size > 0:
                        one_pred = pred[ii, tag_idx]
                        one_image = images[ii]
                        im_shape = one_image.shape[1:]

                        center_coord = np.array(one_image.shape[1:])//2

                        one_pred_r = imresize(one_pred, size=im_shape)
                        max_coord = np.unravel_index(one_pred_r.argmax(),
                                                     one_pred_r.shape)
                        dist_list = compute_euc_distances(one_gt, max_coord)
                        dist = dist_list.min()

                        measure_list.append(dist)
                        in_class_measure_list.append(
                            [song_id, fn_raw, ii, dist]
                        )
                        # raw_input(123)
                    else:
                        continue
        it.write_csv(out_fp, in_class_measure_list)

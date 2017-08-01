#!/usr/bin/env python

import os
import io_tool as it
import numpy as np


def compute_euc_distances(arr, target_vec):
    temp = arr-target_vec
    dist_list = np.sqrt(np.sum(temp**2, axis=1))
    return dist_list


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

    num_segments = 10

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

    # Output
    out_dir = os.path.join(
        base_dir,
        'test_result.spatial.fragment_{}s'.format(fragment_unit),
        'center_baseline')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_fp_1 = os.path.join(out_dir, '{}.micro.csv'.format(measure_type))
    out_fp_2 = os.path.join(out_dir, '{}.classwise.csv'.format(measure_type))

    # Process groundtruth
    print('Process groundtruth...')
    measure_list = list()
    classwise_measure_list = list()
    for jj, tag in enumerate(tag_list):
        tag_idx = tag_list.index(tag)
        id_list = id_dict[tag]

        in_class_measure_list = list()
        for uu, song_id in enumerate(id_list):
            print(uu, song_id)
            gt_dir = os.path.join(base_gt_dir, tag, song_id)

            image_dir = os.path.join(base_image_dir, song_id)

            for fn in test_fn_list:
                image_fp = os.path.join(image_dir, '{}.npy'.format(fn))
                images = np.load(image_fp)
                for ii in range(num_segments):
                    gt_fp = os.path.join(gt_dir, '{}.{}.npy'.format(fn, ii))
                    one_gt = np.load(gt_fp)

                    if one_gt.size > 0:
                        one_image = images[ii]
                        im_shape = one_image.shape[1:]

                        center_coord = np.array(one_image.shape[1:])//2

                        dist_list = compute_euc_distances(one_gt, center_coord)
                        dist = dist_list.min()

                        measure_list.append(dist)
                        in_class_measure_list.append(dist)
                        # raw_input(123)
                    else:
                        continue
        classwise_measure_list.append(np.mean(in_class_measure_list))
    measure = np.mean(measure_list)

    it.write_csv(out_fp_1, [[measure]])
    it.write_csv(out_fp_2, zip(tag_list, classwise_measure_list))

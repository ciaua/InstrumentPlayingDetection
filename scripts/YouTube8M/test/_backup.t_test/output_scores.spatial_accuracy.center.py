#!/usr/bin/env python

import os
import io_tool as it
import numpy as np


def compute_euc_distances(arr, target_vec):
    temp = arr-target_vec
    dist_list = np.sqrt(np.sum(temp**2, axis=1))
    return dist_list


if __name__ == '__main__':
    fragment_unit = 5  # second

    measure_type = 'euclidean_dist'

    test_fn_list = ['0_5', '5_10', '30_35', '35_40']

    phase = 'te'

    # Settings
    param_type = 'best_measure'

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
        'test_result.distance.spatial.fragment_{}s'.format(fragment_unit),
        param_type,
        'center'
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

                image_fp = os.path.join(image_dir, '{}.npy'.format(fn))
                images = np.load(image_fp)
                num_segments = images.shape[0]
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
                        in_class_measure_list.append(
                            [song_id, fn_raw, ii, dist]
                        )
                        # raw_input(123)
                    else:
                        continue
        it.write_csv(out_fp, in_class_measure_list)

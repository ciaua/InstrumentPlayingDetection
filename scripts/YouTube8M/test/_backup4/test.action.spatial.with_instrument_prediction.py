#!/usr/bin/env python

import os
import io_tool as it
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
# from scipy.misc import imresize

# from sklearn.metrics import roc_auc_score, average_precision_score


def upscale(images, new_size, method):
    if method == 'bilinear':
        upsample = nn.UpsamplingBilinear2d(size=new_size)
    if method == 'nearest':
        upsample = nn.UpsamplingNearest2d(size=new_size)
    im = Variable(torch.FloatTensor(images))
    new_im = upsample(im).data.numpy()

    return new_im


def compute_euc_distances(arr, target_vec):
    temp = arr-target_vec
    dist_list = np.sqrt(np.sum(temp**2, axis=1))
    return dist_list


def get_prediction(song_id, sub_time_range_str, base_pred_dir, base_inst_dir):
    # pred_dir_i = os.path.join(base_pred_dir_i, song_id)
    pred_dir = os.path.join(base_pred_dir, song_id)
    inst_dir = os.path.join(base_inst_dir, song_id)

    # pred_fp_i = os.path.join(pred_dir_i,
    #                          '{}.npy'.format(sub_time_range_str))
    pred_fp = os.path.join(pred_dir, '{}.npy'.format(sub_time_range_str))
    inst_fp = os.path.join(inst_dir, '{}.npy'.format(sub_time_range_str))
    # pred_i = np.load(pred_fp_i)
    pred = np.load(pred_fp)
    inst = np.load(inst_fp)

    # size = pred.shape[2:]
    # inst = upscale(inst, size, method=upscale_method)
    # print(pred.shape, inst.shape)
    # raw_input(123)

    pred = (pred*inst)
    return pred


if __name__ == '__main__':
    time_range = (0, 60)
    fragment_unit = 5  # second

    measure_type = 'pixel_euclidean_dist'

    up_method = 'bilinear'
    # upscale_method = 'nearest'

    test_fn_list = ['0_5', '5_10', '30_35', '35_40']

    phase = 'te'

    # Settings
    model_id_fill_list = [
        ('20170709_161154', 1),
        ('20170708_073900', 2),
        ('20170708_154229', 4),
        ('20170711_080438', 8),
        ('20170711_175508', 16),
        ('20170713_153618', 4),  # 0.1 threshold
        ('20170713_002039', 4),  # 0.5 threshold
        ('20170713_055026', 4),  # 0.9 threshold
        ('20170713_224322', 4),  # 0.5 threshold with instrument mask
        ('20170714_144445', 4),  # instrument*audio as target
        ('20170714_155252', 4),  # instrument*audio as target v2
    ]

    inst_model_id = '20170710_211725'

    param_type = 'best_measure'
    # param_type = 'best_loss'
    sr = 16000
    hop = 512

    # Dirs and fps
    base_dir = "/home/ciaua/NAS/home/data/TMM17_instrument_playing/YT8M/"
    base_data_dir = "/home/ciaua/NAS/home/data/youtube8m/"
    id_dir = os.path.join(base_data_dir, 'picked_id')
    id_dict_fp = os.path.join(id_dir, 'picked_id.{}.json'.format(phase))
    id_dict = it.read_json(id_dict_fp)

    base_image_dir = os.path.join(
        base_data_dir,
        'feature.5s_fragment',
        'video.time_0_to_60',
        'image.16000_512.16_frames_per_seg.h_w_max_256')

    # Tag
    tag_fp = os.path.join(base_data_dir, 'tag_list.instrument.csv')
    tag_list = [term[0] for term in it.read_csv(tag_fp)]

    all_id_list = list()
    for tag in tag_list:
        all_id_list += id_dict[tag]

    # Groundtruth and prediction
    base_gt_dir = os.path.join(
        base_data_dir,
        'manual_annotation.action_location.h_w.down_right',
        phase)

    base_inst_dir = os.path.join(
        base_dir,
        'predictions.instrument.no_resize',
        'rgb_image.{}_{}.fragment_{}s'.format(
            time_range[0], time_range[1], fragment_unit),
        inst_model_id, param_type)

    for model_id_fill in model_id_fill_list:
        print(model_id_fill)
        model_id, fill_factor = model_id_fill

        base_pred_dir = os.path.join(
            base_dir,
            'predictions.action.no_resize',
            'dense_optical_flow.fill_{}.{}_{}.fragment_{}s'.format(
                fill_factor, time_range[0], time_range[1], fragment_unit),
            model_id, param_type)

        # Output
        out_dir_avg = os.path.join(
            base_dir,
            'test_result.label_wise.spatial.{}'.format(up_method),
            param_type, '{}__{}'.format(model_id, inst_model_id),
        )
        if not os.path.exists(out_dir_avg):
            os.makedirs(out_dir_avg)
        out_fp_1 = os.path.join(
            out_dir_avg, '{}.micro.csv'.format(measure_type))
        out_fp_2 = os.path.join(
            out_dir_avg, '{}.classwise.csv'.format(measure_type))

        out_dir_video = os.path.join(
            base_dir,
            'test_result.video_wise.spatial.{}'.format(up_method),
            param_type, '{}__{}'.format(model_id, inst_model_id), measure_type)
        if not os.path.exists(out_dir_video):
            os.makedirs(out_dir_video)

        # Process groundtruth
        print('Process groundtruth...')
        measure_list = list()
        classwise_measure_list = list()
        for jj, tag in enumerate(tag_list):
            tag_idx = tag_list.index(tag)
            id_list = id_dict[tag]

            out_fp_video = os.path.join(out_dir_video, '{}.csv'.format(tag))

            video_measure_list = list()
            all_seg_measure_list = list()
            for uu, song_id in enumerate(id_list):
                print(tag, uu, song_id)
                gt_dir = os.path.join(base_gt_dir, tag, song_id)
                image_dir = os.path.join(base_image_dir, song_id)

                seg_measure_list = list()
                for fn in test_fn_list:
                    pred = get_prediction(
                        song_id, fn, base_pred_dir, base_inst_dir)
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

                            one_pred = upscale(
                                one_pred[None, None, :],
                                im_shape, up_method)[0, 0]
                            # one_pred_r = imresize(one_pred, size=im_shape)
                            max_coord = np.unravel_index(
                                one_pred.argmax(), one_pred.shape)
                            dist_list = compute_euc_distances(one_gt, max_coord)
                            dist = dist_list.min()

                            measure_list.append(dist)
                            seg_measure_list.append(dist)
                            # raw_input(123)
                        else:
                            continue
                video_measure = np.mean(seg_measure_list)
                all_seg_measure_list += seg_measure_list
                video_measure_list.append([song_id, video_measure])
            it.write_csv(out_fp_video, video_measure_list)

            classwise_measure = np.mean(all_seg_measure_list)
            classwise_measure_list.append(classwise_measure)
        measure = np.mean(measure_list)

        it.write_csv(out_fp_1, [[measure]])
        it.write_csv(out_fp_2, zip(tag_list, classwise_measure_list))

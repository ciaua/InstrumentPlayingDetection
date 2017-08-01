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

    num_tags = 9

    # Settings
    model_id_fill_list = [
        # ('20170709_161154', 1),
        # ('20170708_073900', 2),
        # ('20170708_154229', 4),
        # ('20170711_080438', 8),
        # ('20170711_175508', 16),
        # ('20170717_222549', 4),  # conv6 rf1
        # ('20170717_222602', 4),  # conv6 rf5
        # ('20170713_153618', 4),  # 0.1 threshold
        # ('20170717_104045', 4),  # 0.3 threshold
        # ('20170713_002039', 4),  # 0.5 threshold
        # ('20170717_103859', 4),  # 0.7 threshold
        # ('20170713_055026', 4),  # 0.9 threshold
        # ('20170713_224322', 4),  # 0.5 threshold with instrument mask
        # ('20170714_144445', 4),  # instrument*audio as target v1, 0.2
        ('20170715_232957', 4),  # inst*audio as target v1, 0.3
        # ('20170716_085223', 4),  # inst*audio as target v1, 0.4
        # ('20170715_132615', 4),  # inst*audio as target v1, 0.5
        # ('20170716_185749', 4),  # inst*audio as target v1, 0.6
        # ('20170716_085231', 4),  # inst*audio as target v1, 0.7
        # ('20170716_185757', 4),  # inst*audio as target v1, 0.8
        # ('20170715_233010', 4),  # inst*audio as target v1, 0.9
        # ('20170714_155252', 4),  # instrument*audio as target v2
        # ('20170715_004352', 4),  # instrument*audio as target v3
        # ('20170722_002530', 4),  # inst as target 0.1 thres
        # ('20170719_155122', 4),  # inst as target 0.3 thres
        # ('20170722_002420', 4),  # inst as target 0.5 thres
        # ('20170722_105620', 4),  # inst as target 0.7 thres
        # ('20170722_105631', 4),  # inst as target 0.9 thres
        # ('20170720_113125', 4),  # Audio model with YT8M 0.5 thres
        # ('20170720_113612', 4),  # Audio model with Magna 0.5 thres
        # ('20170720_205529', 4),  # Audio model with Magna 0.3 thres
        # ('20170720_232406', 4),  # Audio model with Magna 0.1 thres
    ]

    # Instrument model
    inst_model_id = '20170710_211725'
    sound_model_id = '20170710_110230'

    param_type = 'best_measure'

    # Dirs and fps
    base_dir = "/home/ciaua/NAS/home/data/TMM17_instrument_playing/YT8M/"
    base_data_dir = "/home/ciaua/NAS/home/data/youtube8m/"
    db2_dir = "/home/ciaua/NAS/Database2/YouTube8M/"
    id_dir = os.path.join(base_data_dir, 'picked_id')
    id_dict_fp = os.path.join(id_dir, 'picked_id.{}.json'.format(phase))
    id_dict = it.read_json(id_dict_fp)

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

    base_inst_dir = os.path.join(
        base_dir,
        'predictions.instrument.no_resize',
        'rgb_image.{}_{}.fragment_{}s'.format(
            time_range[0], time_range[1], fragment_unit),
        inst_model_id, param_type)

    sr = 16000
    hop = 512
    base_sound_dir = os.path.join(
        db2_dir, 'feature.{}s_fragment'.format(fragment_unit),
        'audio.time_{}_to_{}'.format(*time_range),
        'audio_prediction.{}_{}.{}'.format(sr, hop, sound_model_id))

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
            'test_result.label_wise.temporal.no_resize',
            param_type, '{}__{}__{}'.format(
                model_id, inst_model_id, sound_model_id),
        )
        if not os.path.exists(out_dir_avg):
            os.makedirs(out_dir_avg)

        base_out_dir_video = os.path.join(
            base_dir,
            'test_result.video_wise.temporal.no_resize',
            param_type, '{}__{}__{}'.format(
                model_id, inst_model_id, sound_model_id))

        # Process prediction
        print('Process prediction...')
        pred_dict = dict()
        for uu, song_id in enumerate(all_id_list):
            print(uu, song_id)
            pred_dir = os.path.join(base_pred_dir, song_id)
            inst_dir = os.path.join(base_inst_dir, song_id)
            sound_dir = os.path.join(base_sound_dir, song_id)

            pred_list = list()
            for fn in test_fn_list:
                pred_fp = os.path.join(pred_dir, '{}.npy'.format(fn))
                inst_fp = os.path.join(inst_dir, '{}.npy'.format(fn))
                sound_fp = os.path.join(sound_dir, '{}.npy'.format(fn))

                pred = np.load(pred_fp)
                inst = np.load(inst_fp)
                so = np.load(sound_fp)

                num_segments = pred.shape[0]
                # raw_input(123)

                pred = (pred*inst*so[:, :, None, None]).max(axis=2).max(axis=2)
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
            out_fp_1 = os.path.join(
                out_dir_avg, '{}.micro.csv'.format(measure_type))
            out_fp_2 = os.path.join(
                out_dir_avg, '{}.classwise.csv'.format(measure_type))

            out_dir_video = os.path.join(base_out_dir_video, measure_type)
            if not os.path.exists(out_dir_video):
                os.makedirs(out_dir_video)

            classwise_measure_list = list()
            for ii, tag in enumerate(tag_list):
                print(measure_type, tag)
                id_list = gt_dict[tag]

                out_fp_video = os.path.join(out_dir_video, '{}.csv'.format(tag))

                video_measure_list = list()
                for id_ in id_list:
                    g = gt_dict[tag][id_][:, ii:ii+1]
                    p = pred_dict[id_][:, ii:ii+1]
                    try:
                        if measure_type == 'auc_classwise':
                            measure = roc_auc_score(g, p, None)
                        elif measure_type == 'ap_classwise':
                            measure = average_precision_score(g, p, None)
                        if measure != np.nan:
                            video_measure_list.append((id_, measure))

                        print(id_, measure)
                    except Exception as e:
                        continue
                print(tag)
                it.write_csv(out_fp_video, video_measure_list)

                classwise_measure = np.mean(
                    [term[1] for term in video_measure_list])
                classwise_measure_list.append(classwise_measure)

            measure = np.mean(classwise_measure_list)

            it.write_csv(out_fp_1, [[measure]])
            it.write_csv(out_fp_2, zip(tag_list, classwise_measure_list))

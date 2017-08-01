import os
from jjtorch import utils
import numpy as np
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
    time_range = (0, 60)
    fragment_unit = 5  # second

    measure_type = 'auc_classwise'

    test_fn_list = ['0_5', '5_10', '30_35', '35_40']
    num_tags = 9

    fill_factor = 4

    phase = 'te'

    # Dirs and fps
    # base_data_dir = "/home/ciaua/NAS/home/data/youtube8m/"
    base_data_dir = 'path/to/data/dir'

    id_dict_fp = '../../data/video_id.te.json'
    tag_fp = '../../data/tag_list.instrument.csv'

    # Tag
    tag_list = [term[0] for term in utils.read_csv(tag_fp)]

    id_dict = utils.read_json(id_dict_fp)
    all_id_list = list()
    for tag in tag_list:
        all_id_list += id_dict[tag]

    # Groundtruth and prediction
    base_gt_dir = os.path.join(
        base_data_dir,
        'manual_annotation.action_location.h_w.down_right',
        phase)

    base_pred_dir = os.path.join(
        base_data_dir,
        'predictions.action.no_resize',
        'dense_optical_flow.fill_{}.{}_{}.fragment_{}s'.format(
            fill_factor, time_range[0], time_range[1], fragment_unit))

    # Output
    out_dir_avg = os.path.join(
        base_data_dir,
        'test_result.label_wise.temporal.no_resize',
    )
    if not os.path.exists(out_dir_avg):
        os.makedirs(out_dir_avg)

    base_out_dir_video = os.path.join(
        base_data_dir,
        'test_result.video_wise.temporal.no_resize')

    # Process prediction
    print('Process prediction...')
    pred_dict = dict()
    for uu, song_id in enumerate(all_id_list):
        print(uu, song_id)
        pred_dir = os.path.join(base_pred_dir, song_id)

        pred_list = list()
        for fn in test_fn_list:
            pred_fp = os.path.join(pred_dir, '{}.npy'.format(fn))
            pred = np.load(pred_fp)
            # print(fn)

            num_segments = pred.shape[0]

            pred = pred.max(axis=2).max(axis=2)
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
                if measure != np.nan:
                    video_measure_list.append((id_, measure))

                print(id_, measure)
            except Exception as e:
                continue
        print(tag)
        utils.write_csv(out_fp_video, video_measure_list)

        classwise_measure = np.mean(
            [term[1] for term in video_measure_list])
        classwise_measure_list.append(classwise_measure)

    measure = np.mean(classwise_measure_list)

    utils.write_csv(out_fp_1, [[measure]])
    utils.write_csv(out_fp_2, zip(tag_list, classwise_measure_list))

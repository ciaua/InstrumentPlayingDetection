#!/usr/bin/env python

import os
import numpy as np
import io_tool as it

if __name__ == '__main__':
    fragment_unit = 5  # second

    measure_type_list = ['ap_classwise', 'auc_classwise']

    # Settings
    # model_id_fill = ('20170709_161154', 1)
    # model_id_fill = ('20170708_073900', 2)
    # model_id_fill = ('20170708_154229', 4)
    # model_id_fill = ('20170711_080438', 8)
    # model_id_fill = ('20170711_175508', 16)

    # model_id_fill = ('20170713_153618', 4)  # 0.1 threshold
    # model_id_fill = ('20170713_002039', 4)  # 0.5 threshold
    # model_id_fill = ('20170713_055026', 4)  # 0.9 threshold

    # model_id_fill = ('20170713_224322', 4)  # 0.5 threshold with instrument mask

    # model_id_fill = ('20170714_144445', 4)  # instrument*audio as target
    # model_id_fill = ('20170714_155252', 4)  # instrument*audio as target v2

    # With instrument
    # model_id_fill = ('20170709_161154__20170710_211725', 1)
    # model_id_fill = ('20170708_073900__20170710_211725', 2)
    # model_id_fill = ('20170708_154229__20170710_211725', 4)
    # model_id_fill = ('20170711_080438__20170710_211725', 8)
    # model_id_fill = ('20170711_175508__20170710_211725', 16)

    # model_id_fill = ('20170713_153618__20170710_211725', 4)  # 0.1 threshold
    # model_id_fill = ('20170713_002039__20170710_211725', 4)  # 0.5 threshold
    # model_id_fill = ('20170713_055026__20170710_211725', 4)  # 0.9 threshold

    # model_id_fill = ('20170713_224322__20170710_211725', 4)  # 0.5 threshold with instrument mask

    # model_id_fill = ('20170714_144445__20170710_211725', 4)  # instrument*audio as target
    model_id_fill = ('20170714_155252__20170710_211725', 4)  # instrument*audio as target v2

    param_type = 'best_measure'

    model_id, fill_factor = model_id_fill

    # Dirs and fps
    base_dir = "/home/ciaua/NAS/home/data/TMM17_instrument_playing/YT8M/"
    base_data_dir = "/home/ciaua/NAS/home/data/youtube8m/"

    # Tag
    tag_fp = os.path.join(base_data_dir, 'tag_list.instrument.csv')
    tag_list = [term[0] for term in it.read_csv(tag_fp)]

    # Input
    base_in_dir = os.path.join(
        base_dir,
        'test_result.video_wise.temporal.fill_{}.fragment_{}s.no_resize'.format(
            fill_factor, fragment_unit),
        param_type,
        model_id)

    out_dir = os.path.join(
        base_dir,
        'test_result.label_wise.temporal.fill_{}.fragment_{}s.no_resize'.format(
            fill_factor, fragment_unit),
        param_type,
        model_id)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Process groundtruth
    for measure_type in measure_type_list:
        print(measure_type)
        out_fp = os.path.join(out_dir, '{}.csv'.format(measure_type))
        in_dir = os.path.join(base_in_dir, measure_type)

        out_list = []
        for ii, tag in enumerate(tag_list):
            in_fp = os.path.join(in_dir, '{}.csv'.format(tag))

            score = [float(term[1])
                     for term in it.read_csv(in_fp) if term[1] != 'nan']
            out_list.append([tag, np.mean(score)])

            print(tag)
        it.write_csv(out_fp, out_list)

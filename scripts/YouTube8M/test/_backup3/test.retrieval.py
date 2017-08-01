#!/usr/bin/env python

import os
import io_tool as it


if __name__ == '__main__':
    time_range = (0, 60)
    fragment_unit = 5  # second

    # score_name = 'without_object_without_playing'
    score_name = 'with_object_without_playing'
    # score_name = 'instrument_playing'

    precision_at = 10

    phase_list = ['va', 'te']

    model_id_obj = '20170319_085641'
    model_id_act = '20170403_183041'  # audioanno

    param_type = 'best_measure'

    # Dirs and fps
    base_dir = "/home/ciaua/NAS/home/data/youtube8m/"

    id_dir = os.path.join(base_dir, 'fold.tr16804_va2100_te2100')
    id_list = list()
    for phase in phase_list:
        id_fp = os.path.join(id_dir, 'fold.{}.txt'.format(phase))
        id_list += it.read_lines(id_fp)

    base_in_dir_obj = os.path.join(
        base_dir,
        'predictions_without_resize',
        'rgb_image.{}_{}.fragment_{}s'.format(
            time_range[0], time_range[1], fragment_unit),
        model_id_obj, param_type)
    base_in_dir_act = os.path.join(
        base_dir,
        'predictions_without_resize',
        'dense_optical_flow.{}_{}.fragment_{}s'.format(
            time_range[0], time_range[1], fragment_unit),
        model_id_act, param_type)

    sorted_dir = os.path.join(
        base_dir, 'sorted_videos',
        '{}.{}.{}__{}'.format(score_name, '_'.join(phase_list),
                              model_id_obj, model_id_act))

    # Output
    out_dir = os.path.join(base_dir, 'test.retrieval',
                           'precision@{}.{}__{}'.format(
                               precision_at, model_id_obj, model_id_act))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_fp = os.path.join(out_dir, '{}.csv'.format(score_name))

    # Tag
    tag_fp = os.path.join(base_dir, 'tag_list.instrument.csv')
    tag_list = [term[0] for term in it.read_csv(tag_fp)]

    out_list = list()
    for tag in tag_list:
        print(tag)
        in_fp = os.path.join(sorted_dir, '{}.csv'.format(tag))
        data = it.read_csv(in_fp)[:precision_at]

        r = sum([int(term[-1]) for term in data])
        n = len(data)

        out_list.append((tag, r/float(n)))
    it.write_csv(out_fp, out_list)

import os
import numpy as np
import io_tool as it
import tensorflow as tf


def read_example(example):
    result = tf.train.Example.FromString(example)

    video_id = list(result.features.feature['video_id'].bytes_list.value)[0]
    labels = list(result.features.feature['labels'].int64_list.value)

    return labels, video_id


def do_one(example, out_dir, tag_set, tag_dict,
           count_num, video_limit, num_tags):

    labels, video_id = read_example(example)

    intersection = list(tag_set.intersection(labels))

    if not intersection:
        pass
    else:
        ii_list = [tag_dict[term] for term in intersection]
        if np.all(count_num[ii_list] >= video_limit):
            pass
        else:
            out_fn = '{}.npy'.format(video_id)
            out_fp = os.path.join(out_dir, out_fn)

            vec = np.zeros(num_tags)
            vec[ii_list] = 1
            np.save(out_fp, vec)

            count_num += vec
            # print('Done: {}'.format(video_id))
    return count_num


base_dir = '/home/ciaua/NAS/home/data/youtube8m/'

video_limit_dict = {
    'train': 5000,
    'validate': 1000}

base_in_dir = os.path.join(base_dir, 'Source')
base_out_dir = os.path.join(base_dir, 'annotation')

tag_fp = os.path.join(base_in_dir, 'tag_list.instrument.csv')
tag_list = [int(term[1]) for term in it.read_csv(tag_fp)]
tag_set = set(tag_list)

tag_max_num = np.array([int(term[2]) for term in it.read_csv(tag_fp)])

tag_dict = {int(idx): ii
            for ii, [tag, idx, mn] in enumerate(it.read_csv(tag_fp))}

num_tags = len(tag_list)

for phase in ['train', 'validate']:
    print(phase)
    in_dir = os.path.join(base_in_dir, 'yt8m_video_level', phase)
    fn_list = os.listdir(in_dir)

    video_limit = video_limit_dict[phase]
    out_dir = os.path.join(base_out_dir, '{}.{}'.format(phase, video_limit))

    if os.path.exists(out_dir):
        continue
    else:
        os.makedirs(out_dir)

    count_num = np.zeros((num_tags,))

    for fn in sorted(fn_list):
        if fn in ['1_video_level_train_download_plan.json',
                  '1_video_level_validate_download_plan.json']:
            continue
        print(fn)
        in_fp = os.path.join(in_dir, fn)
        args_list = list()
        for example in tf.python_io.tf_record_iterator(in_fp):
            count_num = do_one(example, out_dir, tag_set, tag_dict,
                               count_num, video_limit, num_tags)

        if np.all(count_num >= np.minimum(tag_max_num, video_limit)):
            break
        print(count_num)

import os
import io_tool as it
import tensorflow as tf
from multiprocessing import Pool


def read_example(example):
    result = tf.train.Example.FromString(example)

    video_id = list(result.features.feature['video_id'].bytes_list.value)[0]
    labels = list(result.features.feature['labels'].int64_list.value)

    return labels, video_id


def do_one(args):
    example, out_dir, tag_set = args

    labels, video_id = read_example(example)

    if not tag_set.intersection(labels):
        return
    else:
        out_fn = '{}.txt'.format(video_id)
        out_fp = os.path.join(out_dir, out_fn)

        it.write_lines(out_fp, labels)
        print('Done: {}'.format(video_id))


base_dir = '/home/ciaua/NAS/home/data/youtube8m/'

num_processes = 40

base_in_dir = os.path.join(base_dir, 'Source')
base_out_dir = os.path.join(base_dir, 'annotation.raw')

tag_fp = os.path.join(base_in_dir, 'tag_list.instrument.csv')
tag_set = set([int(term[1]) for term in it.read_csv(tag_fp)])

pool = Pool(num_processes)

for phase in ['train', 'validate']:
    print(phase)
    in_dir = os.path.join(base_in_dir, 'yt8m_video_level', phase)
    fn_list = os.listdir(in_dir)

    out_dir = os.path.join(base_out_dir, phase)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for fn in fn_list:
        if fn in ['1_video_level_train_download_plan.json',
                  '1_video_level_validate_download_plan.json']:
            continue
        print(fn)
        in_fp = os.path.join(in_dir, fn)
        args_list = list()
        for example in tf.python_io.tf_record_iterator(in_fp):
            args = (example, out_dir, tag_set)
            args_list.append(args)

        print('start one file')
        pool.map(do_one, args_list)
        print('finish one file')

import os
import io_tool as it
import numpy as np


if __name__ == '__main__':
    # ### Options ###
    # dataset = 'Kinetics'
    label_type = 'instrument'
    clip_limit = 5000

    base_dir = '/home/ciaua/NAS/home/data/AudioSet/'

    in_dir = os.path.join(base_dir, 'song_list.{}'.format(label_type))

    base_out_dir = os.path.join(base_dir, 'annotation.{}'.format(label_type))

    # ### Label list ###
    label_fp = os.path.join(base_dir, 'tag_list.{}.csv'.format(label_type))
    label_list = [term[1] for term in it.read_csv(label_fp)]

    # ### Main ###
    for phase in ['tr', 'va']:
        if phase == 'tr':
            in_fn = 'unbalanced_train_segments.{}.csv'.format(clip_limit)
        elif phase == 'va':
            in_fn = 'eval_segments.all.csv'
        in_fp = os.path.join(in_dir, in_fn)
        out_dir = os.path.join(base_out_dir, phase)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        count = 0
        info_list = it.read_csv(in_fp)
        for info in info_list:
            count += 1
            print(count)
            id_ = info[0]
            anno_list = info[3:]
            out_fp = os.path.join(out_dir, '{}.npy'.format(id_))
            if os.path.exists(out_fp):
                print('Processed before')
                continue

            out_vec = np.zeros(len(label_list))

            for anno in anno_list:
                if anno in label_list:
                    label_idx = label_list.index(anno)
                    out_vec[label_idx] = 1

            np.save(out_fp, out_vec)

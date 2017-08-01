import os
import numpy as np

batch_size = 20

base_dir = '/home/ciaua/NAS/home/data/youtube8m/'

# subdir = 'validate.500'
subdir = 'train.2000'

in_dir = os.path.join(base_dir,
                      'annotation', subdir)
out_dir = os.path.join(base_dir,
                       'batch_annotation.{}'.format(batch_size), subdir)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

fn_list = os.listdir(in_dir)
for fn in fn_list:
    in_fp = os.path.join(in_dir, fn)
    out_fp = os.path.join(out_dir, fn)

    anno = np.load(in_fp)[None, :]
    batch_anno = np.repeat(anno, batch_size, axis=0)
    np.save(out_fp, batch_anno)
    print(out_fp)

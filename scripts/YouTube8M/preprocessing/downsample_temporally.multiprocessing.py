import os
import numpy as np
from multiprocessing import Pool


def do_one(args):
    in_fp, out_fp, nn = args
    feat = np.load(in_fp)
    hop = int(np.ceil(feat.shape[0]/float(nn)))

    out_feat = feat[::hop]

    np.save(out_fp, out_feat)
    print(out_fp)


num_cores = 30
base_dir = '/home/ciaua/NAS/home/data/youtube8m/'

# gtype = 'audio.time_30_to_60'
# feat_type = 'binary_temporal_instrument.16000_2048.20170312_112549'

gtype = 'video.time_30_to_60'

feat_type = 'image.16000_512.16_frames_per_seg.h256_w256'
# feat_type = 'dense_optical_flow.16000_512.fill_2.16_frames_per_seg.5_flows_per_frame.h256_w256'

down_to_num_segments = 10

nn = down_to_num_segments

in_dir = os.path.join(base_dir,
                      'feature',
                      gtype, feat_type)
out_dir = os.path.join(base_dir,
                       'feature.downsample_to_{}'.format(down_to_num_segments),
                       gtype, feat_type)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

fn_list = os.listdir(in_dir)

args_list = list()
for fn in fn_list:
    in_fp = os.path.join(in_dir, fn)
    out_fp = os.path.join(out_dir, fn)

    args = (in_fp, out_fp, nn)
    args_list.append(args)

pool = Pool(num_cores)

pool.map(do_one, args_list)

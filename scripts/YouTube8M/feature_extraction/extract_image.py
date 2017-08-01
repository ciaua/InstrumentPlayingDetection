import os
import numpy as np
import moviepy.editor as mpy
from moviepy.video import fx

from multiprocessing import Pool


def extract_one(vid,
                sr, hop, time_range, target_size,
                num_frames_per_seg):

    fps = sr/float(hop*num_frames_per_seg)

    # resize
    width, height = vid.size
    factor = min(target_size[0]/float(height), target_size[1]/float(width))
    new_height = round(height*factor)
    new_width = round(width*factor)

    vid = vid.resize(height=new_height, width=new_width)
    new_width, new_height = vid.size

    # pad
    pad_height, pad_width = target_size-(new_height, new_width)
    vid = fx.all.margin(vid, bottom=int(pad_height), right=int(pad_width))

    # shape=(frames, height, width, RGB_channels)
    images = np.stack(vid.iter_frames(fps=fps))

    # shape=(frames, RGB_channels, height, width)
    images = np.transpose(images, [0, 3, 1, 2]).astype('uint8')
    return images


def do_one(args):
    out_dir, vid_dir, fn, vid_ext, \
        sr, hop, time_range, target_size, \
        num_frames_per_seg = args

    # output
    youtube_id = fn.replace(vid_ext, '')
    out_fp = os.path.join(out_dir, '{}.npy'.format(youtube_id))
    if os.path.exists(out_fp):
        print('Done before: {}'.format(fn))
        return

    vid_fp = os.path.join(vid_dir, fn)

    vid = mpy.VideoFileClip(vid_fp)
    if vid.duration < time_range[1]:
        print('Too short: {}.'.format(fn))
        return

    vid = vid.subclip(*time_range)

    try:
        output = extract_one(
            vid, sr, hop, time_range, target_size,
            num_frames_per_seg)
        if tuple(output.shape[2:]) != tuple(target_size):
            print(
                'Wrong Size: {}. {}, {}'.format(fn,
                                                tuple(output.shape[2:]),
                                                tuple(target_size)))
            return
        np.save(out_fp, output)
        print('Done: {}'.format(fn))

    except Exception as e:
        print('Error: {}. {}'.format(fn, repr(e)))
        return


if __name__ == '__main__':

    # Settings
    num_cores = 20
    vid_dir = '/home/ciaua/NAS/home/data/youtube8m/video/'
    base_feat_dir = '/home/ciaua/NAS/home/data/youtube8m/feature/'
    vid_ext = '.mp4'

    sr = 16000
    hop = 512

    num_frames_per_seg = 8

    target_size = np.array((256, 256))  # (height, width)

    time_range = (30, 60)

    feat_type = '.'.join([
        'image',
        '{}_{}'.format(sr, hop),
        '{}_frames_per_seg'.format(num_frames_per_seg),
        'h{}_w{}'.format(target_size[0], target_size[1])
    ])

    # Misc
    fn_list = os.listdir(vid_dir)

    out_dir = os.path.join(
        base_feat_dir,
        'video.time_{}_to_{}'.format(*time_range),
        feat_type
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    args_list = list()
    for fn in fn_list:
        args = (out_dir, vid_dir, fn, vid_ext,
                sr, hop, time_range, target_size,
                num_frames_per_seg)
        args_list.append(args)

    pool = Pool(num_cores)
    pool.map(do_one, args_list)

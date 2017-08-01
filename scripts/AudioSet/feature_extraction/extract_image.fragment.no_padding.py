import os
import io_tool as it
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
    factor = min(target_size/float(height), target_size/float(width))
    new_height = round(height*factor)
    new_width = round(width*factor)

    vid = vid.resize(height=new_height, width=new_width)
    new_width, new_height = vid.size

    # pad
    pad_long_side = target_size-max(new_height, new_width)
    if new_height >= new_width:
        vid = fx.all.margin(vid, bottom=int(pad_long_side))
    else:
        vid = fx.all.margin(vid, right=int(pad_long_side))

    # shape=(frames, height, width, RGB_channels)
    images = np.stack(vid.iter_frames(fps=fps))

    # shape=(frames, RGB_channels, height, width)
    images = np.transpose(images, [0, 3, 1, 2]).astype('uint8')
    return images


def do_one(args):
    base_out_dir, vid_dir, fn, vid_ext, \
        sr, hop, time_range, target_size, \
        num_frames_per_seg, fragment_unit = args

    # output
    youtube_id = fn.replace(vid_ext, '')
    out_dir = os.path.join(base_out_dir, youtube_id)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    vid_fp = os.path.join(vid_dir, fn)
    if not os.path.exists(vid_fp):
        print('No video: {}.'.format(fn))
        return

    vid = mpy.VideoFileClip(vid_fp)
    if vid.duration < time_range[1]:
        print('Too short: {}.'.format(fn))
        return

    num_fragments = (time_range[1]-time_range[0]) // fragment_unit

    for ii in range(num_fragments):

        sub_time_range = (ii*fragment_unit, (ii+1)*fragment_unit)
        out_fp = os.path.join(out_dir, '{}_{}.npy'.format(*sub_time_range))
        if os.path.exists(out_fp):
            print('Done before: {}. Fragment {}'.format(fn, ii))
            return

        sub_vid = vid.subclip(*sub_time_range)

        try:
            output = extract_one(
                sub_vid, sr, hop, time_range, target_size,
                num_frames_per_seg)
            if np.max(output.shape[2:]) != target_size:
                print(
                    'Wrong Size: {}. Fragment {}. {}, {}'.format(
                        fn, ii,
                        tuple(output.shape[2:]),
                        target_size))
                return
            np.save(out_fp, output)

        except Exception as e:
            print('Error: {}. Fragment {}. {}'.format(fn, ii, repr(e)))
            continue
    print('Done: {}'.format(fn))


if __name__ == '__main__':

    # Settings
    bbase_feat_dir = '/home/ciaua/NAS/home/data/AudioSet/'
    num_cores = 10
    vid_dir = '/home/ciaua/NAS/Database2/AudioSet/video/'
    vid_ext = '.mp4'

    sr = 16000
    hop = 512

    fold_dir = '/home/ciaua/NAS/home/data/AudioSet/song_list/'
    fn_fp_list = [os.path.join(fold_dir, fn) for fn in os.listdir(fold_dir)]

    num_frames_per_seg = 16

    target_size = 256  # maximum(h, w)

    # time_range = (0, 60)
    fragment_unit = 5  # second

    base_feat_dir = os.path.join(
        bbase_feat_dir,
        'feature.{}s_fragment'.format(fragment_unit))

    feat_type = '.'.join([
        'image',
        '{}_{}'.format(sr, hop),
        '{}_frames_per_seg'.format(num_frames_per_seg),
        'h_w_max_{}'.format(target_size)
    ])

    # Misc
    # fn_list = os.listdir(vid_dir)
    info_list = list()
    for fn_fp in fn_fp_list:
        info_list += [('{}{}'.format(term[0], vid_ext), tuple(term[1:3]))
                      for term in it.read_csv(fn_fp)]

    base_out_dir = os.path.join(base_feat_dir, 'video.target_time', feat_type)

    args_list = list()
    for info in info_list:
        fn = info[0]
        time_range = tuple(map(int, map(float, info[1])))
        args = (base_out_dir, vid_dir, fn, vid_ext,
                sr, hop, time_range, target_size,
                num_frames_per_seg, fragment_unit)
        args_list.append(args)

    pool = Pool(num_cores)
    pool.map(do_one, args_list)

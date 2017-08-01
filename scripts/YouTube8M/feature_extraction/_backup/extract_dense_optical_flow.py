import os
import numpy as np
import moviepy.editor as mpy
from moviepy.video import fx
import cv2

from multiprocessing import Pool


def extract_one(vid,
                sr, hop, time_range, target_size,
                num_frames_per_seg, num_flows_per_frame, fill_factor):
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

    fps = sr/float(hop*num_frames_per_seg)
    num_frames = int(round((time_range[1]-time_range[0])*fps))

    fake_fps = fill_factor*fps
    sub_frames = np.stack(list(vid.iter_frames(fake_fps))).astype('uint8')

    half_num_flows_per_frame = (num_flows_per_frame-1)/2

    # shape=(num_padded_sub_frames, height, width, 3)
    sub_frames = np.pad(
        sub_frames,
        pad_width=((half_num_flows_per_frame+1, half_num_flows_per_frame),
                   (0, 0), (0, 0), (0, 0)), mode='edge')

    # sub_frames = zoom(sub_frames, (1, factor, factor, 1))

    # frame1 = im_iterator.next()
    # prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    flow_list = list()
    for ii in range(sub_frames.shape[0]-1):
        frame1 = sub_frames[ii]
        frame2 = sub_frames[ii+1]

        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_list.append(flow)
        # print(ii)
        # print(flow.max())

    # shape=(num_sub_frames+num_flows_per_frame-1, height, width, 2)
    flows = np.stack(flow_list)

    stacked_flow_list = list()
    for ii in range(num_frames):
        idx_begin = ii*fill_factor
        idx_end = idx_begin + num_flows_per_frame

        # shape=(num_flows_per_frame, height, width, 2)
        stacked_flow = flows[idx_begin: idx_end]

        # shape=(num_flows_per_frame, 2, height, width)
        stacked_flow = np.transpose(stacked_flow, axes=(0, 3, 1, 2))

        stacked_flow_list.append(stacked_flow)

    # shape=(num_frames, num_flows_per_frame, 2, height, width)
    stacked_flow_all = np.stack(stacked_flow_list)

    shape = stacked_flow_all.shape

    stacked_flow_all = np.reshape(
        stacked_flow_all, (shape[0], -1, shape[3], shape[4]))

    # scale it up
    # stacked_flow_all *= 2

    stacked_flow_all = stacked_flow_all.astype('uint8')
    return stacked_flow_all


def do_one(args):
    out_dir, vid_dir, fn, vid_ext, \
        sr, hop, time_range, target_size, \
        num_frames_per_seg, num_flows_per_frame, fill_factor = args

    # output
    youtube_id = fn.replace(vid_ext, '')
    out_fp = os.path.join(out_dir, '{}.npy'.format(youtube_id))
    if os.path.exists(out_fp):
        print('Done before: {}'.format(fn))
        return

    vid_fp = os.path.join(vid_dir, fn)

    vid = mpy.VideoFileClip(vid_fp)
    vid = vid.subclip(*time_range)

    try:
        output = extract_one(
            vid, sr, hop, time_range, target_size,
            num_frames_per_seg, num_flows_per_frame, fill_factor)
        if np.array(output.shape[2:]) != np.array(target_size):
            print(
                'Wrong Size: {} ...................................'.format(fn))
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

    num_frames_per_seg = 16

    target_size = np.array((256, 256))  # (height, width)

    fill_factor = 2  # sample with the rate fill_factor*fps to fill the gap

    # include the neighboring flows:
    # left (num_flows-1)/2, self, and right (num_flows-1)/2
    # Must be an odd number
    num_flows_per_frame = 5
    time_range = (30, 60)

    feat_type = '.'.join([
        'dense_optical_flow',
        '{}_{}'.format(sr, hop),
        'fill_{}'.format(fill_factor),
        '{}_frames_per_seg'.format(num_frames_per_seg),
        '{}_flows_per_frame'.format(num_flows_per_frame),
        'h{}_w{}'.format(target_size[0], target_size[1])
    ])

    assert(num_flows_per_frame % 2 == 1)

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
                num_frames_per_seg, num_flows_per_frame, fill_factor)
        args_list.append(args)

    pool = Pool(num_cores)
    pool.map(do_one, args_list)

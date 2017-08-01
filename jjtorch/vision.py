import numpy as np
import moviepy.editor as mpy
import cv2


def extract_images(vid, fps):

    # Frames per second
    # fps = sr/float(hop*num_frames_per_seg)

    # shape=(frames, height, width, RGB_channels)
    images = np.stack(vid.iter_frames(fps=fps))

    # shape=(frames, RGB_channels, height, width)
    images = np.transpose(images, [0, 3, 1, 2]).astype('uint8')
    return images


def extract_dense_optical_flows(vid, real_fps, dof_fps, num_flows_per_frame):
    '''
    dof_fps:
        determining the resolution of fps
    '''

    real_spf = 1./real_fps
    dof_spf = 1./dof_fps

    duration = vid.duration

    num_frames = int(round(duration*real_fps))

    half_num_flows_per_frame = (num_flows_per_frame-1)/2

    stacked_flow_list = list()
    for ii in range(num_frames):
        # print(ii, num_frames)
        # print(num_frames)
        center_time = ii*real_spf + real_spf/2
        start_time = center_time - dof_spf*(half_num_flows_per_frame+1)
        end_time = start_time+dof_spf*(num_flows_per_frame+1)
        if start_time > duration:
            break

        sub_vid = vid.subclip(start_time, end_time)
        sub_frames = np.stack(
            list(sub_vid.iter_frames(dof_fps))).astype('uint8')
        # shape=(num_sub_frames, height, width, 3)
        # print(sub_frames.shape)

        if sub_frames.shape[0] >= (num_flows_per_frame+1):
            sub_frames = sub_frames[:num_flows_per_frame+1]
        else:
            total_pad = num_flows_per_frame+1-sub_frames.shape[0]
            pad_left = (total_pad+1)//2
            pad_right = total_pad-pad_left
            sub_frames = np.pad(
                sub_frames,
                pad_width=(
                    (pad_left, pad_right), (0, 0), (0, 0), (0, 0)), mode='edge')

        flow_list = list()
        for jj in range(num_flows_per_frame):
            frame1 = sub_frames[jj]
            frame2 = sub_frames[jj+1]

            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_list.append(flow)
        stacked_flow = np.stack(flow_list)

        # shape=(num_flows_per_frame, height, width, 2)
        # stacked_flow = flows[idx_begin:idx_end]

        # shape=(num_flows_per_frame, 2, height, width)
        stacked_flow = np.transpose(stacked_flow, axes=(0, 3, 1, 2))

        stacked_flow_list.append(stacked_flow)

    # shape=(num_frames, num_flows_per_frame, 2, height, width)
    stacked_flow_all = np.stack(stacked_flow_list)

    shape = stacked_flow_all.shape

    stacked_flow_all = np.reshape(
        stacked_flow_all, (shape[0], -1, shape[3], shape[4]))

    stacked_flow_all = np.minimum(np.maximum(stacked_flow_all, -128), 127)

    print(stacked_flow_all.shape)
    return stacked_flow_all


def extract_one_dense_optical_flow(args):
    idx, video_fp, time_range, real_fps, dof_fps, num_flows_per_frame = args
    vid = mpy.VideoFileClip(video_fp)
    if time_range is not None:
        vid = vid.subclip(*time_range)

    real_spf = 1./real_fps
    dof_spf = 1./dof_fps
    duration = vid.duration
    half_num_flows_per_frame = (num_flows_per_frame-1)/2

    center_time = idx*real_spf + real_spf/2
    start_time = center_time - dof_spf*(half_num_flows_per_frame+1)
    end_time = start_time+dof_spf*(num_flows_per_frame+1)
    if start_time > duration:
        return

    sub_vid = vid.subclip(start_time, end_time)
    sub_frames = np.stack(
        list(sub_vid.iter_frames(dof_fps))).astype('uint8')
    # shape=(num_sub_frames, height, width, 3)
    # print(sub_frames.shape)

    if sub_frames.shape[0] >= (num_flows_per_frame+1):
        sub_frames = sub_frames[:num_flows_per_frame+1]
    else:
        total_pad = num_flows_per_frame+1-sub_frames.shape[0]
        pad_left = (total_pad+1)//2
        pad_right = total_pad-pad_left
        sub_frames = np.pad(
            sub_frames,
            pad_width=(
                (pad_left, pad_right), (0, 0), (0, 0), (0, 0)), mode='edge')

    flow_list = list()
    for jj in range(num_flows_per_frame):
        frame1 = sub_frames[jj]
        frame2 = sub_frames[jj+1]

        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_list.append(flow)
    stacked_flow = np.stack(flow_list)

    # shape=(num_flows_per_frame, height, width, 2)
    # stacked_flow = flows[idx_begin:idx_end]

    # shape=(num_flows_per_frame, 2, height, width)
    stacked_flow = np.transpose(stacked_flow, axes=(0, 3, 1, 2))
    return stacked_flow


def extract_dense_optical_flows_mp(
        video_fp, time_range,
        real_fps, dof_fps, num_flows_per_frame, num_cores=5):
    '''
    multiprocessing

    dof_fps:
        determining the resolution of fps
    '''
    vid = mpy.VideoFileClip(video_fp)
    if time_range is not None:
        vid = vid.subclip(*time_range)

    duration = vid.duration

    num_frames = int(round(duration*real_fps))

    args_list = list()
    for ii in range(num_frames):
        # print(ii, num_frames)
        # print(num_frames)
        args = (ii, video_fp, time_range,
                real_fps, dof_fps, num_flows_per_frame)

        args_list.append(args)

    from multiprocessing import Pool
    # from pathos.multiprocessing import ProcessingPool as Pool
    pool = Pool(num_cores)
    stacked_flow_list = pool.map(extract_one_dense_optical_flow, args_list)
    pool.close()

    stacked_flow_list = [term for term in stacked_flow_list if term is not None]

    # shape=(num_frames, num_flows_per_frame, 2, height, width)
    stacked_flow_all = np.stack(stacked_flow_list)

    shape = stacked_flow_all.shape

    stacked_flow_all = np.reshape(
        stacked_flow_all, (shape[0], -1, shape[3], shape[4]))

    stacked_flow_all = np.minimum(np.maximum(stacked_flow_all, -128), 127)

    print(stacked_flow_all.shape)
    return stacked_flow_all


if __name__ == '__main__':
    real_fps = 2*(16000./(512*16))

    fill_factor = 4

    audio_fps = 16000./(512*16)
    dof_fps = audio_fps*fill_factor
    time_range = (0, 10)
    num_flows_per_frame = 5
    num_cores = 5
    # raw_input(123)

    vid_fp = '/home/ciaua/NAS/Database2/YouTube8M/video/7N3jE_4t46E.mp4'
    vid = mpy.VideoFileClip(vid_fp)
    sub_vid = vid.subclip(*time_range)
    dof = extract_dense_optical_flows_mp(
        vid_fp, time_range, real_fps, dof_fps, num_flows_per_frame, num_cores)
    imgs = extract_images(sub_vid, real_fps)

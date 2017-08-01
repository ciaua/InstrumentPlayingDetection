#!/usr/bin/env python

import os
import cv2
from jjtorch import utils
import io_tool as it
import numpy as np

from jjtorch.layers import SpatialCrossMapLRN as LRN
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable

import moviepy.editor as mpy
# from moviepy.video import fx

gid = 0  # GPU id
torch.cuda.set_device(gid)


class Net(nn.Module):
    def __init__(self, num_labels, feat_dim):
        super(Net, self).__init__()

        # Basic
        self.num_labels = num_labels

        # Common
        self.dropout = nn.Dropout(p=0.5)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

        #
        self.conv1 = nn.Conv2d(feat_dim, 96, kernel_size=7, stride=2, padding=3)
        self.lrn1 = LRN(5, alpha=1e-4, k=2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv6 = nn.Conv2d(512, 2048, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0)
        self.conv8 = nn.Conv2d(
            1024, num_labels, kernel_size=1, stride=1, padding=0)

        # Initialization
        init.xavier_uniform(self.conv1.weight)
        init.xavier_uniform(self.conv2.weight)
        init.xavier_uniform(self.conv3.weight)
        init.xavier_uniform(self.conv4.weight)
        init.xavier_uniform(self.conv5.weight)
        init.xavier_uniform(self.conv6.weight)
        init.xavier_uniform(self.conv7.weight)
        init.xavier_uniform(self.conv8.weight)
        self.conv1.bias.data.zero_()
        self.conv2.bias.data.zero_()
        self.conv3.bias.data.zero_()
        self.conv4.bias.data.zero_()
        self.conv5.bias.data.zero_()
        self.conv6.bias.data.zero_()
        self.conv7.bias.data.zero_()
        self.conv8.bias.data.zero_()

    def forward(self, x):
        # Input: x, shape=(batch_size, feat_dim, num_frames, 1)

        # Early
        x = self.pool(self.lrn1(F.relu(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))

        # Late
        x = self.dropout(F.relu(self.conv6(x)))
        x = self.dropout(F.relu(self.conv7(x)))
        x = F.sigmoid(self.conv8(x))

        pooled_x = F.max_pool2d(x, kernel_size=x.size()[2:]).view(x.size()[:2])

        return x, pooled_x


def get_video_handler(video_fp, time_range=None):
    vid = mpy.VideoFileClip(video_fp)

    if time_range is not None:
        vid = vid.subclip(*time_range)

    return vid


def extract_one(vid,
                sr, hop, time_range, target_size,
                num_frames_per_seg, num_flows_per_frame, fill_factor):
    # resize
    width, height = vid.size
    if target_size is None:
        pass
    else:
        factor = min(target_size/float(height), target_size/float(width))
        new_height = round(height*factor)
        new_width = round(width*factor)

        vid = vid.resize(height=new_height, width=new_width)
        # new_width, new_height = vid.size

    real_fps = sr/float(hop*num_frames_per_seg)
    num_frames = int(round((time_range[1]-time_range[0])*real_fps))

    # fake_fps = fill_factor*fps
    # fill_factor = hop/fps
    temp_fps = real_fps*fill_factor
    finer_frames = np.stack(list(vid.iter_frames(temp_fps))).astype('uint8')

    half_num_flows_per_frame = (num_flows_per_frame-1)/2

    # shape=(num_padded_sub_frames, height, width, 3)
    finer_frames = np.pad(
        finer_frames,
        pad_width=((half_num_flows_per_frame+1, half_num_flows_per_frame),
                   (0, 0), (0, 0), (0, 0)), mode='edge')
    # print(finer_frames.shape)

    # sub_frames = zoom(sub_frames, (1, factor, factor, 1))

    # frame1 = im_iterator.next()
    # prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    """
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
    """

    # shape=(num_sub_frames+num_flows_per_frame-1, height, width, 2)
    # print(flows.shape)

    stacked_flow_list = list()
    for ii in range(num_frames):
        idx_mid = ii*fill_factor+fill_factor//2+half_num_flows_per_frame
        if fill_factor == 1:
            idx_begin = idx_mid-half_num_flows_per_frame
        else:
            idx_begin = idx_mid-half_num_flows_per_frame-1
        idx_end = idx_begin + num_flows_per_frame+1

        sub_frames = finer_frames[idx_begin:idx_end]

        # print(idx_begin, idx_end)
        # raw_input(123)
        flow_list = list()
        for ii in range(num_flows_per_frame):
            frame1 = sub_frames[ii]
            frame2 = sub_frames[ii+1]

            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_list.append(flow)
        stacked_flow = np.stack(flow_list)
        # print(stacked_flow.shape)
        # raw_input(123)

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

    # scale it up
    # stacked_flow_all *= 2

    stacked_flow_all = np.minimum(np.maximum(stacked_flow_all, -128), 127)

    print(stacked_flow_all.shape)
    return stacked_flow_all


if __name__ == '__main__':
    sub_time_range_list = [(0, 5), (5, 10), (30, 35), (35, 40)]
    time_range = (0, 60)
    fragment_unit = 5  # second
    num_fragments = (time_range[1]-time_range[0]) // fragment_unit

    phase = 'te'

    # Settings
    sr = 16000
    hop = 512
    num_tags = 9

    # DO NOT CHANGE: dense optical flow option
    # fill_factor = 2  # sample with the rate fill_factor*fps to fill the gap
    # include the neighboring flows:
    # left (num_flows-1)/2, self, and right (num_flows-1)/2
    # Must be an odd number
    num_flows_per_frame = 5

    num_frames_per_seg = 16

    # model_id_fill = ('20170709_161154', 1)
    # model_id_fill = ('20170708_073900', 2)
    # model_id_fill = ('20170708_154229', 4)
    # model_id_fill = ('20170711_080438', 8)

    # ## Audio target
    model_id_fill = ('20170713_153618', 4)  # 0.1 threshold
    # model_id_fill = ('20170713_002039', 4)  # 0.5 threshold
    # model_id_fill = ('20170713_055026', 4)  # 0.9 threshold

    # model_id_fill = ('20170713_224322', 4)  # 0.5 threshold with instrument mask

    max_hw = 256  # max(height, width)
    target_size = None

    param_type = 'best_measure'
    # param_type = 'best_loss'

    # Dirs and fps
    base_dir = "/home/ciaua/NAS/home/data/TMM17_instrument_playing/YT8M/"
    base_data_dir = "/home/ciaua/NAS/home/data/youtube8m/"
    db2_dir = "/home/ciaua/NAS/Database2/YouTube8M/"
    id_dir = os.path.join(base_data_dir, 'picked_id')
    video_dir = os.path.join(db2_dir, 'video')
    id_dict_fp = os.path.join(id_dir, 'picked_id.{}.json'.format(phase))
    id_dict = it.read_json(id_dict_fp)

    model_id_o, fill_factor = model_id_fill

    id_list = list()
    for inst in id_dict:
        id_list += id_dict[inst]

    # Output
    base_out_dir = os.path.join(
        base_dir,
        'predictions.action.max_hw_{}'.format(max_hw),
        'dense_optical_flow.fill_{}.{}_{}.fragment_{}s'.format(
            fill_factor, time_range[0], time_range[1], fragment_unit),
        model_id_o, param_type)

    # Dirs and fps
    save_dir = os.path.join(base_dir, 'save.action')
    model_dir_o = os.path.join(save_dir, model_id_o)

    # Model: Network structure
    print('Making network...')
    net = Net(num_tags, feat_dim=10)
    net.eval()
    net.cuda()

    # Load params
    param_fp_i = os.path.join(
        save_dir, model_id_o, 'model', 'params.{}.torch'.format(param_type))
    utils.load_model(param_fp_i, net)

    for uu, song_id in enumerate(id_list):
        print(uu, song_id)
        # output fp
        out_dir = os.path.join(base_out_dir, song_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Load video
        video_fp = os.path.join(video_dir, '{}.mp4'.format(song_id))
        vid = get_video_handler(video_fp, None)

        width, height = vid.size
        factor = min(max_hw/float(height), max_hw/float(width))
        new_height = int(round(height*factor))
        new_width = int(round(width*factor))

        for ff in range(num_fragments):
            sub_time_range = (ff*fragment_unit, (ff+1)*fragment_unit)
            if sub_time_range not in sub_time_range_list:
                continue
            out_fp = os.path.join(out_dir, '{}_{}.npy'.format(*sub_time_range))
            if os.path.exists(out_fp):
                print('Done before')
                continue
            # Extract dense optical flow
            print('Extract optical flow...')
            sub_vid = vid.subclip(*sub_time_range)
            dof = extract_one(
                sub_vid, sr, hop, sub_time_range, target_size,
                num_frames_per_seg, num_flows_per_frame, fill_factor)

            # Predict
            print('Predict...')

            pred_list_o = list()
            for one_dof in dof:
                one_dof = \
                    Variable(
                        torch.FloatTensor(one_dof.astype('float32'))).cuda()
                pred_one_o, _ = net(one_dof[None, :])
                upscale = nn.UpsamplingBilinear2d(size=(new_height, new_width))
                pred_one_o_up = upscale(pred_one_o)

                pred_one_o_up = pred_one_o_up.data.cpu().numpy()

                pred_list_o.append(pred_one_o_up)
            pred_o = np.concatenate(pred_list_o, axis=0)

            np.save(out_fp, pred_o)

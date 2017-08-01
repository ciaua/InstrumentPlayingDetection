#!/usr/bin/env python

import os
import io_tool as it
import numpy as np

from jjtorch import utils
from jjtorch.layers import SpatialCrossMapLRN as LRN
# from multiprocessing import Pool

import torch
# from torchnet.dataset import ListDataset
# from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

import moviepy.editor as mpy
# from moviepy.video import fx
# import cv2

gid = 0  # GPU id
torch.cuda.set_device(gid)


def get_video_handler(video_fp, time_range=None):
    vid = mpy.VideoFileClip(video_fp)

    if time_range is not None:
        vid = vid.subclip(*time_range)

    return vid


def extract_images(vid,
                   sr, hop, time_range, num_frames_per_seg):

    # Frames per second
    fps = sr/float(hop*num_frames_per_seg)

    # shape=(frames, height, width, RGB_channels)
    images = np.stack(vid.iter_frames(fps=fps))

    # shape=(frames, RGB_channels, height, width)
    images = np.transpose(images, [0, 3, 1, 2]).astype('uint8')
    return images


class Net(nn.Module):
    def __init__(self, num_labels, feat_dim, feat_mean):
        super(Net, self).__init__()

        # Basic
        self.mean = Variable(
            torch.FloatTensor(feat_mean[None, :, None, None]).cuda())
        # torch.FloatTensor(feat_mean[None, :, None, None]))

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

        self.conv6 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0)
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

        # Standardization
        x = (x-self.mean.expand_as(x))

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


if __name__ == '__main__':
    sub_time_range_list = [(0, 5), (5, 10), (30, 35), (35, 40)]
    time_range = (0, 60)
    fragment_unit = 5  # second
    num_fragments = (time_range[1]-time_range[0]) // fragment_unit

    phase = 'te'

    # Settings
    sr = 16000
    hop = 512
    num_labels = 9

    num_frames_per_seg = 16
    # target_size = None  # (height, width)

    model_id_i = '20170711_234822'

    param_type = 'best_measure'
    # param_type = 'best_loss'

    # Dirs and fps
    base_dir = "/home/ciaua/NAS/home/data/TMM17_instrument_playing/YT8M/"
    db2_dir = "/home/ciaua/NAS/Database2/YouTube8M/"
    base_data_dir = "/home/ciaua/NAS/home/data/youtube8m/"
    id_dir = os.path.join(base_data_dir, 'picked_id')
    video_dir = os.path.join(db2_dir, 'video')
    id_dict_fp = os.path.join(id_dir, 'picked_id.{}.json'.format(phase))
    id_dict = it.read_json(id_dict_fp)

    id_list = list()
    for inst in id_dict:
        id_list += id_dict[inst]

    # Output
    base_out_dir = os.path.join(
        base_dir,
        'predictions.instrument.no_resize',
        'rgb_image.{}_{}.fragment_{}s'.format(
            time_range[0], time_range[1], fragment_unit),
        model_id_i, param_type)

    # Dirs and fps
    save_dir = os.path.join(base_dir, 'save.image')
    model_dir_i = os.path.join(save_dir, model_id_i)

    # Load the scaler
    pretrained_model_dir = os.path.join(base_data_dir, 'pretrained_models')
    pretrained_model_fp = os.path.join(pretrained_model_dir,
                                       'FCN.VGG_CNN_M_2048.RGB.pytorch')
    pmodel = torch.load(pretrained_model_fp)
    mean_image = pmodel['mean_image'].mean(axis=0).mean(axis=0)

    # Model: Network structure
    print('Making network...')
    net = Net(num_labels, feat_dim=3, feat_mean=mean_image)
    net.eval()
    net.cuda()

    # Load params
    param_fp_i = os.path.join(
        save_dir, model_id_i, 'model', 'params.{}.torch'.format(param_type))
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

        for ff in range(num_fragments):
            sub_time_range = (ff*fragment_unit, (ff+1)*fragment_unit)
            if sub_time_range not in sub_time_range_list:
                continue

            out_fp = os.path.join(out_dir, '{}_{}.npy'.format(*sub_time_range))
            if os.path.exists(out_fp):
                print('Done before')
                continue

            # Extract dense optical flow
            print('Extract images...')
            sub_vid = vid.subclip(*sub_time_range)
            images = extract_images(
                sub_vid, sr, hop, sub_time_range,
                num_frames_per_seg)

            # Predict
            print('Predict...')

            pred_list_i = list()
            for one_image in images:
                one_image = Variable(
                    torch.FloatTensor(one_image.astype('float32'))).cuda()
                pred_one, _ = net(one_image[None, :])
                pred_one = pred_one.data.cpu().numpy()

                pred_list_i.append(pred_one)
            pred_i = np.concatenate(pred_list_i, axis=0)

            np.save(out_fp, pred_i)

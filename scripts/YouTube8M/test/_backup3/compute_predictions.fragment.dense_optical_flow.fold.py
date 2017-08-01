#!/usr/bin/env python

import os
# import lasagne
from lasagne import layers
from jj import utils
# import jj.layers as cl
import io_tool as it
import numpy as np
import theano
import theano.tensor as T

from lasagne.layers import InputLayer
# from lasagne.layers import ReshapeLayer
from lasagne.layers import GlobalPoolLayer
# from lasagne.layers import DimshuffleLayer
# from lasagne.layers import DenseLayer
# from lasagne.layers import NonlinearityLayer
# from lasagne.layers import ExpressionLayer
# from lasagne.layers import ElemwiseMergeLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
# from lasagne.nonlinearities import softmax, sigmoid, rectify
from lasagne.nonlinearities import sigmoid
# from lasagne.utils import floatX as to_floatX

import moviepy.editor as mpy
# from moviepy.video import fx
import cv2

ConvLayer = Conv2DLayer
floatX = theano.config.floatX


def make_structure_image(input_var, num_tags):
    mean_image = np.array([123.50936127, 115.7726059, 102.71698761])
    mean_image = mean_image.reshape((1, 3, 1, 1)).astype(floatX)

    net = {}
    input_var = input_var - mean_image  # RGB
    net['input'] = InputLayer((None, 3, None, None), input_var)
    net['conv1'] = ConvLayer(net['input'],
                             num_filters=96,
                             filter_size=7,
                             stride=2,
                             pad=3,
                             flip_filters=False)
    # caffe has alpha = alpha * pool_size
    net['norm1'] = NormLayer(net['conv1'], alpha=0.0001)
    net['pool1'] = PoolLayer(net['norm1'],
                             pool_size=3,
                             stride=2,
                             pad=(1, 1),
                             ignore_border=True)
    net['conv2'] = ConvLayer(net['pool1'],
                             num_filters=256,
                             filter_size=5,
                             stride=2,
                             pad=2,
                             flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2'],
                             pool_size=3,
                             stride=2,
                             pad=(1, 1),
                             ignore_border=True)
    net['conv3'] = ConvLayer(net['pool2'],
                             num_filters=512,
                             filter_size=3,
                             pad=1,
                             flip_filters=False)
    net['conv4'] = ConvLayer(net['conv3'],
                             num_filters=512,
                             filter_size=3,
                             pad=1,
                             flip_filters=False)
    net['conv5'] = ConvLayer(net['conv4'],
                             num_filters=512,
                             filter_size=3,
                             pad=1,
                             flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5'],
                             pool_size=3,
                             stride=2,
                             pad=(1, 1),
                             ignore_border=True)

    # Late conv
    net['conv6'] = ConvLayer(net['pool5'],
                             num_filters=2048,
                             filter_size=3,
                             pad=1)  # change to pad=1 when testing
    net['drop6'] = DropoutLayer(net['conv6'],
                                p=0.5)
    net['conv7'] = ConvLayer(net['drop6'],
                             num_filters=1024,
                             filter_size=1)
    net['drop7'] = DropoutLayer(net['conv7'],
                                p=0.5)
    net['conv8'] = ConvLayer(net['drop7'],
                             num_filters=num_tags,
                             filter_size=1,
                             nonlinearity=sigmoid)
    net['pooled_output'] = GlobalPoolLayer(net['conv8'],
                                           pool_function=T.max)

    return net


def make_structure_optical_flow(input_var, num_tags):

    net = {}
    net['input'] = InputLayer((None, 10, None, None), input_var)
    net['conv1'] = ConvLayer(net['input'],
                             num_filters=96,
                             filter_size=7,
                             stride=2,
                             pad=3,
                             flip_filters=False)
    # caffe has alpha = alpha * pool_size
    net['norm1'] = NormLayer(net['conv1'], alpha=0.0001)
    net['pool1'] = PoolLayer(net['norm1'],
                             pool_size=3,
                             stride=2,
                             pad=(1, 1),
                             ignore_border=True)
    net['conv2'] = ConvLayer(net['pool1'],
                             num_filters=256,
                             filter_size=5,
                             stride=2,
                             pad=2,
                             flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2'],
                             pool_size=3,
                             stride=2,
                             pad=(1, 1),
                             ignore_border=True)
    net['conv3'] = ConvLayer(net['pool2'],
                             num_filters=512,
                             filter_size=3,
                             pad=1,
                             flip_filters=False)
    net['conv4'] = ConvLayer(net['conv3'],
                             num_filters=512,
                             filter_size=3,
                             pad=1,
                             flip_filters=False)
    net['conv5'] = ConvLayer(net['conv4'],
                             num_filters=512,
                             filter_size=3,
                             pad=1,
                             flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5'],
                             pool_size=3,
                             stride=2,
                             pad=(1, 1),
                             ignore_border=True)

    # Late conv
    net['conv6'] = ConvLayer(net['pool5'],
                             num_filters=2048,
                             filter_size=3,
                             pad=1)  # change to pad=1 when testing
    net['drop6'] = DropoutLayer(net['conv6'],
                                p=0.5)
    net['conv7'] = ConvLayer(net['drop6'],
                             num_filters=1024,
                             filter_size=1)
    net['drop7'] = DropoutLayer(net['conv7'],
                                p=0.5)
    net['conv8'] = ConvLayer(net['drop7'],
                             num_filters=num_tags,
                             filter_size=1,
                             nonlinearity=sigmoid)
    net['pooled_output'] = GlobalPoolLayer(net['conv8'],
                                           pool_function=T.max)

    return net


def get_video_handler(video_fp, time_range=None):
    vid = mpy.VideoFileClip(video_fp)

    if time_range is not None:
        vid = vid.subclip(*time_range)

    return vid


def extract_dense_optical_flows(vid,
                                sr, hop, time_range,
                                num_frames_per_seg, num_flows_per_frame,
                                fill_factor):
    # Frames per second
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

    stacked_flow_all = np.minimum(np.maximum(stacked_flow_all, -127), 128)

    return stacked_flow_all


if __name__ == '__main__':
    time_range = (0, 60)
    fragment_unit = 5  # second
    num_fragments = (time_range[1]-time_range[0]) // fragment_unit

    # phase = 'te'
    phase = 'va'

    # Settings
    sr = 16000
    hop = 512
    num_tags = 9

    # DO NOT CHANGE: dense optical flow option
    fill_factor = 2  # sample with the rate fill_factor*fps to fill the gap
    # include the neighboring flows:
    # left (num_flows-1)/2, self, and right (num_flows-1)/2
    # Must be an odd number
    num_flows_per_frame = 5

    num_frames_per_seg = 16

    model_id_o = '20170403_183405'  # anno
    # model_id_o = '20170403_183041'  # audioanno

    param_type = 'best_measure'
    # param_type = 'best_loss'

    # Dirs and fps
    base_dir = "/home/ciaua/NAS/home/data/youtube8m/"
    video_dir = os.path.join(base_dir, 'video')

    id_dir = os.path.join(base_dir, 'fold.tr16804_va2100_te2100')
    id_fp = os.path.join(id_dir, 'fold.{}.txt'.format(phase))
    id_list = it.read_lines(id_fp)

    # Output
    base_out_dir = os.path.join(
        base_dir,
        'predictions_without_resize',
        'dense_optical_flow.{}_{}.fragment_{}s'.format(
            time_range[0], time_range[1], fragment_unit),
        model_id_o, param_type)

    # Tag
    # tag_fp = os.path.join(base_dir, 'tag_list.instrument.csv')
    # tag_list = [term[0] for term in it.read_csv(tag_fp)]

    # anno_dir = os.path.join(base_dir, )

    # fn_list = os.listdir(feat_dir)

    # Dirs and fps
    save_dir = os.path.join(base_dir, 'save.video')
    model_dir_o = os.path.join(save_dir, model_id_o)
    # model_dir_o = os.path.join(save_dir, model_id_o)

    # Model: Network structure
    print('Making network...')
    input_var_o = T.tensor4('visual_input.optical_flow')

    net_dict_o = make_structure_optical_flow(input_var_o, num_tags)

    network_o = net_dict_o['conv8']

    # Load params
    param_fp_o = os.path.join(
        save_dir, model_id_o, 'model', 'params.{}.npz'.format(param_type))
    utils.load_model(param_fp_o, network_o)

    # Compute loss
    output_va_var_o = layers.get_output(network_o, deterministic=True)

    # Make functions
    input_var_list = [input_var_o]
    func_pr = theano.function(input_var_list, output_va_var_o)

    for uu, song_id in enumerate(id_list):
        print(uu, song_id)
        # output fp
        out_dir = os.path.join(base_out_dir, song_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Load video
        video_fp = os.path.join(video_dir, '{}.mp4'.format(song_id))
        try:
            vid = get_video_handler(video_fp, None)
        except Exception as e:
            print('Error: {}. {}'.format(video_fp, repr(e)))
            continue

        for ff in range(num_fragments):
            sub_time_range = (ff*fragment_unit, (ff+1)*fragment_unit)
            out_fp = os.path.join(out_dir,
                                  '{}_{}.npy'.format(*sub_time_range))
            if os.path.exists(out_fp):
                print('Done before')
                continue

            # Extract dense optical flow
            print('Extract optical flow...')
            sub_vid = vid.subclip(*sub_time_range)
            dof = extract_dense_optical_flows(
                sub_vid, sr, hop, sub_time_range,
                num_frames_per_seg, num_flows_per_frame, fill_factor)

            # Predict
            print('Predict...')

            pred_list_o = list()
            for one_dof in dof:
                pred_one_o = func_pr(one_dof[None, :])
                pred_list_o.append(pred_one_o)
            pred_o = np.concatenate(pred_list_o, axis=0)

            np.save(out_fp, pred_o)

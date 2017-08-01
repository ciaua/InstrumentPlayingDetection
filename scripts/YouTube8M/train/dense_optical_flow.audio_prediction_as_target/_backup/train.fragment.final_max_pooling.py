#!/usr/bin/env python

import os
import lasagne
from lasagne import layers
from jj import load_data as ld
from jj import utils
# import jj.layers as cl
import numpy as np
import theano
import theano.tensor as T
import io_tool as it

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
ConvLayer = Conv2DLayer

floatX = theano.config.floatX


def make_structure(input_var, num_tags):

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
                             pad=0)  # change to pad=1 when testing
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


if __name__ == '__main__':
    model_type = 'fcn_optical_flow'
    model_id = utils.get_current_time()
    print(model_id)

    # Data options
    base_dir = "/home/ciaua/NAS/home/data/youtube8m/"
    base_out_dir = "/home/ciaua/NAS/home/data/youtube8m/"

    feat_type = '.'.join([
        'dense_optical_flow',
        '16000_512.fill_2.16_frames_per_seg.5_flows_per_frame.h_w_max_256',
        '10x10xD2xD3_to_100xD2*D3_png'
    ])

    time_range = (0, 60)
    fragment_unit = 5  # second
    audioanno_model_id = '20170312_112549'  # 16 frames per seg, pool [4, 4]
    anno_feats_dir = os.path.join(
        base_dir,
        'valid_audioanno_feats.time_{}_to_{}.{}s_fragment.{}.{}'.format(
            time_range[0], time_range[1], fragment_unit,
            audioanno_model_id,
            feat_type))

    num_tags = 9
    feat_type_list = [feat_type]

    # Training options
    measure_type = 'mean_auc_y'
    lr = 0.001
    loss_function = lasagne.objectives.binary_crossentropy
    n_epochs = 100
    batch_size = 10  # not used in this setting

    num_cached = 3

    # Dirs and fps
    save_dir = os.path.join(base_out_dir, 'save.video')
    output_dir = os.path.join(save_dir, model_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    info_fp = os.path.join(output_dir, 'info.pkl')
    structure_fp = os.path.join(output_dir, 'structure.pkl')

    # Loading data
    print("Loading data...")
    anno_feats_tr_fp = os.path.join(anno_feats_dir, 'fp_dict.tr.json')
    anno_feats_va_fp = os.path.join(anno_feats_dir, 'fp_dict.va.json')
    anno_feats_te_fp = os.path.join(anno_feats_dir, 'fp_dict.te.json')

    X_tr_list, y_tr, X_va_list, y_va, X_te_list, y_te = \
        ld.load_by_file_fragment(
            anno_feats_tr_fp,
            anno_feats_va_fp,
            anno_feats_te_fp,
        )
    # raw_input(123)

    # Model: Network structure
    print('Making network...')
    target_var = T.matrix('targets')
    input_var = T.tensor4('visual_input')
    input_var_list = [input_var]

    net_dict = make_structure(input_var, num_tags)
    network = net_dict['pooled_output']

    # Compute loss
    lr_var = theano.shared(np.array(lr, dtype=floatX))
    output_var = layers.get_output(network)
    epsilon = np.float32(1e-6)
    one = np.float32(1)
    output_var = T.clip(output_var, epsilon, one-epsilon)
    loss_var = loss_function(output_var, target_var)
    loss_var = loss_var.mean()

    params = layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.momentum(
        loss_var, params, learning_rate=lr_var, momentum=0.9)

    output_va_var = layers.get_output(network, deterministic=True)
    output_va_var = T.clip(output_va_var, epsilon, one-epsilon)
    loss_va_var = loss_function(output_va_var, target_var)
    loss_va_var = loss_va_var.mean()

    # Save model structure
    it.pickle(structure_fp, network)

    # Make functions
    func_tr = theano.function(
        input_var_list+[target_var], loss_var, updates=updates)
    func_va = theano.function(
        input_var_list+[target_var], [output_va_var, loss_va_var])
    func_pr = theano.function(input_var_list, output_va_var)

    # Save info
    save_info = [
        ('model_type', model_type),
        ('model_id', model_id),
        ('loss_function', loss_function),
        ('lr', lr),
        ('measure_type', measure_type),
        ('n_epochs', n_epochs),
        ('batch_size', batch_size),
        ('base_dir', base_dir),
        ('feat_type_list', feat_type_list),
        ('num_tags', num_tags),
    ]

    utils.save_info(info_fp, save_info)

    # Training
    utils.train_by_file_fragment(
        X_tr_list, y_tr, X_va_list, y_va,
        network, measure_type,
        func_tr, func_va,
        n_epochs, batch_size, lr_var,
        X_te_list=X_te_list, y_te=y_te,
        model_id=model_id,
        output_dir=output_dir,
        num_cached=num_cached,
        default_fragment_idx=3
    )

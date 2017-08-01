#!/usr/bin/env python

import os
from jj import utils, measure
import io_tool as it
import numpy as np

import lasagne
from lasagne import layers
# from lasagne.layers import get_all_layers
import theano
import theano.tensor as T


def get_sub_prediction_from_prediction(prediction, label_idx_list_list):
    out_list = list()
    for label_idx_list in label_idx_list_list:
        temp = np.max(prediction[:, label_idx_list], axis=1, keepdims=True)
        out_list.append(temp)

    sub_prediction = np.concatenate(out_list, axis=1)
    return sub_prediction


def make_convs(network,
               num_filters_list, filter_size_list, stride_list, pool_size_list):
    for ii, [num_filters, filter_size, stride, pool_size] in enumerate(
            zip(num_filters_list, filter_size_list,
                stride_list, pool_size_list)):
        network = layers.Conv2DLayer(
            network, num_filters,
            filter_size, stride,
            pad='same',
            name='conv.{}'.format(ii))
        network = layers.MaxPool2DLayer(
            network, (pool_size, 1), ignore_border=False,
            name='maxpool.{}'.format(ii))
    return network


def structure(scaler_list):
    network_list = list()
    input_var_list = list()

    num_tags = 9

    feat_size = 128
    num_filters_list = [1024, 1024]
    filter_size_list = [(7, 1), (1, 1)]
    stride_list = [(1, 1), (1, 1)]
    pool_size_list = [4, 4]

    input_var_list = list()
    for ii, scaler in enumerate(scaler_list):
        input_var = T.tensor4('input')
        input_layer = layers.InputLayer(
            shape=(None, 1, None, feat_size),
            input_var=input_var)

        network = input_layer

        network = layers.DimshuffleLayer(
            network, (0, 3, 2, 1))

        network = layers.standardize(network,
                                     scaler.mean_.astype('float32'),
                                     scaler.scale_.astype('float32'))

        network = make_convs(network,
                             num_filters_list,
                             filter_size_list, stride_list, pool_size_list)

        network_list.append(network)
        input_var_list.append(input_var)

    # Concatenate
    network = layers.ConcatLayer(network_list, axis=1,
                                 cropping=[None, None, 'lower', None])

    # Late
    network = layers.Conv2DLayer(
        layers.dropout(network, p=0.5), 512, (1, 1), (1, 1),
        name='d.late_conv_1')

    network = layers.Conv2DLayer(
        layers.dropout(network, p=0.5), 512, (1, 1), (1, 1),
        name='d.late_conv_2')

    network = layers.Conv2DLayer(
        layers.dropout(network, p=0.5), num_tags, (1, 1), (1, 1),
        nonlinearity=lasagne.nonlinearities.sigmoid,
        name='d.late_conv_3')

    network = layers.ReshapeLayer(network, ([0], [1], [2]))
    frame_output_layer = network

    network = layers.GlobalPoolLayer(network, pool_function=T.mean)

    return network, input_var_list, frame_output_layer


def merge_anno(anno, model_tag_list, test_tag_list, c_dict):
    n = len(model_tag_list)
    t = anno.shape[0]

    out_anno = np.zeros((t, n))
    for ii, tag in enumerate(model_tag_list):
        idx_list = [test_tag_list.index(term) for term in c_dict[tag]]
        anno_list = anno[:, idx_list]
        merged_anno = anno_list.max(axis=1)

        out_anno[:, ii] = merged_anno

    return out_anno


if __name__ == '__main__':
    model_id = '20170312_112549'
    pool_size = 16

    param_type = 'best_measure'

    # Options
    sr = 16000
    hop_size = 512
    # test_measure_type = 'mean_auc_y'
    test_measure_type = 'auc_y_classwise'
    # test_measure_type = 'ap_y_classwise'

    num_train_tags = 9

    c_dict = {
        'Accordion': ['accordion'],
        'Cello': ['cello'],
        'Drummer': ['drum set'],
        'Flute': ['flute'],
        'Guitar': ['acoustic guitar',
                   'clean electric guitar',
                   'distorted electric guitar'],
        'Piano': ['piano'],
        'Saxophone': ['baritone saxophone',
                      'soprano saxophone',
                      'tenor saxophone'],
        'Trumpet': ['trumpet'],
        'Violin': ['violin']
    }

    # (for model) Dirs and fps
    model_base_dir = "/home/ciaua/NAS/home/data/youtube8m/"
    base_data_dir = '/home/ciaua/NAS/home/data/youtube8m'
    train_data_dir = '/home/ciaua/NAS/home/data/youtube8m'

    # (for test) Dirs
    test_base_dir = '/home/ciaua/NAS/home/data/MedleyDB/'
    base_out_dir = model_base_dir

    out_dir = os.path.join(base_out_dir, 'test_result.audio.frame.merge',
                           model_id)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Main
    print(model_id)

    # output
    out_fp = os.path.join(out_dir, '{}.csv'.format(test_measure_type))

    # model related
    save_dir = os.path.join(model_base_dir, 'save.audio')
    info_fp = os.path.join(save_dir, model_id, 'info.pkl')
    save_info = it.unpickle(info_fp)
    info = dict(save_info)

    # Default setting
    feat_type_list = info['feat_type_list']
    feat_type_raw_list = [feat_type.replace('.standard', '.raw')
                          for feat_type in feat_type_list]
    # n_top = info['num_tags']
    num_sources = len(feat_type_list)

    # Test data
    feat_dir_list = [os.path.join(test_base_dir, 'feature', feat_type_raw)
                     for feat_type_raw in feat_type_raw_list]
    anno_dir = os.path.join(test_base_dir,
                            'annotation.medleydb_yt8m',
                            '{}_{}'.format(sr, hop_size))

    # Scaler dir
    model_data_dir = os.path.join(
        train_data_dir,
        'exp_data.audio.time_30_to_60')

    scaler_fp_list = [os.path.join(
        model_data_dir,
        feat_type.replace('.standard', '.raw'), 'scaler.pkl')
        for feat_type in feat_type_list]
    scaler_list = [it.unpickle(scaler_fp) for scaler_fp in scaler_fp_list]

    # Network structure
    # network_fp = os.path.join(save_dir, model_id, 'structure.pkl')
    # network = utils.unpickle(network_fp)
    network, input_var_list, frame_output_layer = structure(scaler_list)

    # Load params
    # param_type = 'params.best_measure'
    param_fp = os.path.join(
        save_dir, model_id, 'model', 'params.{}.npz'.format(param_type))
    utils.load_model(param_fp, network)

    # Make functions
    frame_output_va_var = layers.get_output(frame_output_layer,
                                            deterministic=True)

    # file name list
    fn_list = os.listdir(anno_dir)

    # Load tag list
    model_tag_fp = os.path.join(
        base_data_dir,
        'tag_list.instrument.csv')
    model_tag_list = [term[0] for term in it.read_csv(model_tag_fp)]
    # test_tag_fp = os.path.join(
    #     test_base_dir,
    #     'instrument_list.top{}.txt'.format(
    #         num_top_test_tags))
    tag_conv_fp = os.path.join(
        test_base_dir,
        'instrument_list.medleydb_yt8m.csv')

    tag_conv_dict = dict([(term[0], term[1:])
                          for term in it.read_csv(tag_conv_fp)])
    test_tag_list = [term[0] for term in it.read_csv(tag_conv_fp)]

    label_idx_list_list = [
        list(map(model_tag_list.index, tag_conv_dict[tag]))
        for tag in test_tag_list]

    # pred func
    func_pr = theano.function(input_var_list, frame_output_va_var)

    # Predict
    anno_all = None
    pred_all = None
    for fn in fn_list:
        anno_fp = os.path.join(anno_dir, fn)
        feat_fp_list = [os.path.join(feat_dir, fn)
                        for feat_dir in feat_dir_list]

        # Process annotation
        anno_ = np.load(anno_fp)
        anno = merge_anno(anno_, model_tag_list, test_tag_list, c_dict)

        n_frames = anno.shape[0]

        feat_list = [np.load(feat_fp).astype('float32')
                     for feat_fp in feat_fp_list]

        # standardize
        # feat_list = [scaler.transform(feat) for feat, scaler in zip(
        #     feat_list, scaler_list)]

        feat_list = [feat[None, None, :].astype('float32')
                     for feat in feat_list]
        # prediction = func(*feat_list)[0][0].T

        # Predict and upscale
        prediction = func_pr(*feat_list)
        # prediction = prediction[0, :, :, 0].T
        prediction = prediction[0].T

        # upscale
        prediction = np.repeat(prediction, pool_size, axis=0)

        prediction = prediction[:n_frames]

        # Narrow down
        # sub_prediction = get_sub_prediction_from_prediction(
        #     prediction, label_idx_list_list)
        # prediction = prediction[:, label_idx_list]

        try:
            anno_all = np.concatenate([anno_all, anno], axis=0)
            pred_all = np.concatenate([pred_all, prediction], axis=0)
        except:
            anno_all = anno
            pred_all = prediction

        # print(fn, anno.shape, pred_binary.shape)
        # raw_input(123)

    # out_dict = dict()
    measure_func = getattr(measure, test_measure_type)
    test_score = measure_func(anno_all, pred_all)

    it.write_csv(out_fp, zip(model_tag_list, test_score))

    # print("{}:\t\t{:.4f}".format(test_measure_type, test_score))

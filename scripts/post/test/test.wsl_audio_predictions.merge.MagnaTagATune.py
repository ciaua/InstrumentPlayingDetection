#!/usr/bin/env python

import os
from jj import utils, measure
import numpy as np
import theano
from lasagne.layers import get_all_layers
from lasagne import layers
import io_tool as it


def get_sub_prediction_from_prediction(prediction, label_idx_list_list):
    out_list = list()
    for label_idx_list in label_idx_list_list:
        temp = np.max(prediction[:, label_idx_list], axis=1, keepdims=True)
        out_list.append(temp)

    sub_prediction = np.concatenate(out_list, axis=1)
    return sub_prediction


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
    test_exp_type = 'clip2frame_multisource_gaussian'

    model_id = '20160309_111546'

    upscale_method = 'naive'

    # Options
    sr = 16000
    hop_size = 512
    idx_layer = -4
    gaussian_filter_size = 128
    test_measure_type = 'auc_y_classwise'
    n_top_test_tags = 9

    c_dict = {
        'cello': ['cello'],
        'drums': ['drum set'],
        'flute': ['flute'],
        'guitar': ['acoustic guitar',
                   'clean electric guitar',
                   'distorted electric guitar'],
        'piano': ['piano'],
        'sax': ['baritone saxophone',
                'soprano saxophone',
                'tenor saxophone'],
        'trumpet': ['trumpet'],
        'violin': ['violin']
    }

    # (for model) Dirs and fps
    model_base_dir = '/home/ciaua/NAS/home/data/magna/'
    base_data_dir = '/home/ciaua/NAS/home/data/magna/'

    # (for test) Dirs
    test_base_dir = '/home/ciaua/NAS/home/data/MedleyDB/'

    base_out_dir = '/home/ciaua/NAS/home/data/clip2frame/'
    out_dir = os.path.join(base_out_dir,
                           'test_result.no_threshold.auc',
                           'train_with_gaussian.test_without_gaussian')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(model_id)

    # output
    out_fp = os.path.join(out_dir, '{}.csv'.format(model_id))

    # model related
    save_dir = os.path.join(model_base_dir, 'save')
    param_fp = os.path.join(save_dir, model_id, 'model',
                            'params.best.npz')
    info_fp = os.path.join(save_dir, model_id, 'info.pkl')
    save_info = it.unpickle(info_fp)
    info = dict(save_info)

    # Default setting
    feat_type_list = info['feat_type_list']
    feat_type_raw_list = [feat_type.replace('.standard', '.raw')
                          for feat_type in feat_type_list]
    n_top = info['n_top']
    n_sources = len(feat_type_list)

    feat_dir_list = [os.path.join(test_base_dir, 'feature', feat_type_raw)
                     for feat_type_raw in feat_type_raw_list]
    anno_dir = os.path.join(test_base_dir,
                            'annotation.medleydb_magnatagatune.for_yt8m',
                            '{}_{}'.format(sr, hop_size))

    # file name list
    fn_list = os.listdir(anno_dir)

    # Load tag list
    sub_model_tag_list = ['cello', 'drums', 'flute', 'guitar', 'piano',
                          'sax', 'trumpet', 'violin']
    model_tag_fp = os.path.join(base_data_dir,
                                'labels.top{}.txt'.format(n_top))
    model_tag_list = [term[0] for term in it.read_csv(model_tag_fp)]
    # test_tag_fp = os.path.join(
    #     test_base_dir,
    #     'instrument_list.top{}.txt'.format(
    #         n_top_test_tags))
    tag_conv_fp = os.path.join(
        test_base_dir,
        'instrument_list.medleydb_magnatagatune.for_yt8m.csv')

    # test_tag_list = it.read_lines(test_tag_fp)
    tag_conv_dict = dict([(term[0], term[1:])
                          for term in it.read_csv(tag_conv_fp)])

    test_tag_list = [term[0] for term in it.read_csv(tag_conv_fp)]

    label_idx_list = [model_tag_list.index(term) for term in sub_model_tag_list]

    # Scaler dir
    model_data_dir = os.path.join(base_data_dir,
                                  'exp_data', 'top{}'.format(n_top))

    scaler_fp_list = [os.path.join(
        model_data_dir,
        feat_type.replace('.standard', '.raw'), 'scaler.pkl')
        for feat_type in feat_type_list]
    scaler_list = [it.unpickle(scaler_fp) for scaler_fp in scaler_fp_list]

    # Network options
    network_type = info['network_type']
    network_options = info['network_options']
    network, input_var_list = utils.make_sym_network_multisource(
        network_type, n_sources, network_options
    )
    pool_filter_list = \
        network_options['early_conv_dict_list'][0]['pool_filter_list']
    scale_factor = np.prod(pool_filter_list)

    # Load params
    utils.load_model(param_fp, network)

    # Get frame output layer
    layer_list = get_all_layers(network)
    frame_output_layer = layer_list[idx_layer]
    # raw_input(123)

    # Make predicting function
    frame_prediction = layers.get_output(frame_output_layer,
                                         deterministic=True)
    func = theano.function(input_var_list, [frame_prediction])

    print(upscale_method)

    # Predict
    anno_all = None
    pred_all = None
    for fn in fn_list:
        anno_fp = os.path.join(anno_dir, fn)
        feat_fp_list = [os.path.join(feat_dir, fn)
                        for feat_dir in feat_dir_list]

        # Process annotation
        anno_ = np.load(anno_fp)
        anno = merge_anno(anno_, sub_model_tag_list, test_tag_list, c_dict)

        n_frames = anno.shape[0]

        feat_list = [np.load(feat_fp).astype('float32')
                     for feat_fp in feat_fp_list]

        # standardize
        feat_list = [scaler.transform(feat) for feat, scaler in zip(
            feat_list, scaler_list)]

        feat_list = [feat[None, None, :].astype('float32')
                     for feat in feat_list]
        # prediction = func(*feat_list)[0][0].T

        # Predict and upscale
        out_axis = 2
        in_axis = 2
        prediction = utils.upscale(func, feat_list,
                                   upscale_method, scale_factor,
                                   in_axis, out_axis)
        prediction = prediction[0].T

        prediction = prediction[:n_frames]

        # Narrow down
        prediction = prediction[:, label_idx_list]
        # prediction = get_sub_prediction_from_prediction(
        #     prediction, label_idx_list_list)

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
    # print("{}:\t\t{:.4f}".format(test_measure_type, test_score))

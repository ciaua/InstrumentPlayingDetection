import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"

import theano
import theano.tensor as T
import lasagne
from lasagne import layers
import numpy as np
import io_tool as it
from jj import utils
# from multiprocessing import Pool


def do_one(args):
    fn, base_in_feat_dir, base_out_feat_dir, func_pr, threshold, \
        fragment_unit = args

    # Get youtube id
    in_feat_dir = os.path.join(base_in_feat_dir, fn)
    out_feat_dir = os.path.join(base_out_feat_dir, fn)
    if not os.path.exists(out_feat_dir):
        os.makedirs(out_feat_dir)

    fn_list = os.listdir(in_feat_dir)

    for fn in fn_list:
        in_feat_fp = os.path.join(in_feat_dir, fn)
        out_feat_fp = os.path.join(out_feat_dir, fn)

        if os.path.exists(out_feat_fp):
            print('Done before: {}'.format(fn))
            return

        # Extract feature
        try:
            in_feat = np.load(in_feat_fp)[None, None, :].astype('float32')
            out_feat = func_pr(in_feat)
            out_feat = out_feat[0, :, :, 0].T
            out_feat = (out_feat > threshold).astype('int8')
            np.save(out_feat_fp, out_feat)
            print('Done: {}'.format(fn))

        except Exception as e:
            print('Exception in extracting feature: {}. {}'.format(fn, repr(e)))
            return


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


def make_structure(scaler_list, pool_size_list):
    network_list = list()
    input_var_list = list()

    num_tags = 9

    feat_size = 128
    num_filters_list = [1024, 1024]
    filter_size_list = [(7, 1), (1, 1)]
    stride_list = [(1, 1), (1, 1)]

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
        network, 512, (1, 1), (1, 1),
        name='d.late_conv_1')

    network = layers.Conv2DLayer(
        layers.dropout(network, p=0.5), 512, (1, 1), (1, 1),
        name='d.late_conv_2')

    network = layers.Conv2DLayer(
        layers.dropout(network, p=0.5), num_tags, (1, 1), (1, 1),
        nonlinearity=lasagne.nonlinearities.sigmoid,
        name='d.late_conv_3')

    frame_output_layer = network

    network = layers.ReshapeLayer(network, ([0], [1], [2]))

    network = layers.GlobalPoolLayer(network, pool_function=T.mean)

    return input_var_list, network, frame_output_layer


if __name__ == '__main__':
    num_cores = 10

    threshold = 0.9

    model_id = '20170312_112549'  # 16 frames per seg, pool [4, 4]
    # model_id = '20170315_122919'  # 8 frames per seg, pool [4, 2]
    model_time_range = (30, 60)
    pool_size_list = [4, 4]
    param_type = 'best_measure'

    in_feat_type = 'logmelspec10000'
    out_feat_type = 'binary_temporal_instrument'
    sr = 16000
    win_size = 2048
    hop_size = 512
    num_mels = 128

    time_range = (0, 60)
    fragment_unit = 5  # second
    # fragment_unit = 10  # second

    base_dir = '/home/ciaua/NAS/home/data/youtube8m/'
    save_dir = os.path.join(base_dir, 'save.audio')
    model_dir = os.path.join(save_dir, model_id)

    base_in_feat_dir = os.path.join(
        base_dir, 'feature.{}s_fragment'.format(fragment_unit),
        'audio.time_{}_to_{}'.format(*time_range),
        '{}.{}_{}_{}_{}.0.raw'.format(
            in_feat_type, sr, win_size, hop_size, num_mels)
    )

    base_out_feat_dir = os.path.join(
        base_dir, 'feature.{}s_fragment'.format(
            fragment_unit),
        'audio.time_{}_to_{}'.format(*time_range),
        '{}.threshold_{}.{}_{}.{}'.format(
            out_feat_type, str(threshold).replace('.', ''),
            sr, hop_size, model_id)
    )

    # Load the scaler
    expdata_dir = os.path.join(
        base_dir,
        'exp_data.audio.time_{}_to_{}'.format(*model_time_range),
        '{}.{}_{}_{}_{}.0.raw'.format(
            in_feat_type, sr, win_size, hop_size, num_mels)
    )
    scaler_fp = os.path.join(expdata_dir, 'scaler.pkl')
    scaler = it.unpickle(scaler_fp)

    input_var_list, network, frame_output_layer = make_structure(
        [scaler], pool_size_list)

    # make pred func
    output_va_var = layers.get_output(frame_output_layer, deterministic=True)

    # Load param
    param_fp = os.path.join(
        model_dir, 'model', 'params.{}.npz'.format(param_type))
    utils.load_model(param_fp, network)

    # Make functions
    func_pr = theano.function(input_var_list, output_va_var)

    fn_list = os.listdir(base_in_feat_dir)

    args_list = list()
    for fn in fn_list:
        args = (fn, base_in_feat_dir, base_out_feat_dir, func_pr, threshold,
                fragment_unit)
        args_list.append(args)
    # raw_input(123)

    # pool = Pool(processes=num_cores)
    # pool.map(main, args_list)
    map(do_one, args_list)

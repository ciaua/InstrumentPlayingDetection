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
from PIL import Image
from xml.etree import cElementTree as ElementTree
from scipy.misc import imresize

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


class XmlListConfig(list):
    def __init__(self, aList):
        for element in aList:
            if element:
                # treat like dict
                if len(element) == 1 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                # treat like list
                elif element[0].tag == element[1].tag:
                    self.append(XmlListConfig(element))
            elif element.text:
                text = element.text.strip()
                if text:
                    self.append(text)


class XmlDictConfig(dict):
    '''
    Example usage:

    >>> tree = ElementTree.parse('your_file.xml')
    >>> root = tree.getroot()
    >>> xmldict = XmlDictConfig(root)

    Or, if you want to use an XML string:

    >>> root = ElementTree.XML(xml_string)
    >>> xmldict = XmlDictConfig(root)

    And then use xmldict for what it is... a dict.
    '''
    def __init__(self, parent_element):
        if parent_element.items():
            self.update(dict(parent_element.items()))
        for element in parent_element:
            if element:
                # treat like dict - we assume that if the first two tags
                # in a series are different, then they are all different.
                if len(element) == 1 or element[0].tag != element[1].tag:
                    aDict = XmlDictConfig(element)
                # treat like list - we assume that if the first two tags
                # in a series are the same, then the rest are the same.
                else:
                    # here, we put the list in dictionary; the key is the
                    # tag name the list elements all share in common, and
                    # the value is the list itself
                    aDict = {element[0].tag: XmlListConfig(element)}
                # if the tag has attributes, add those to the dict
                if element.items():
                    aDict.update(dict(element.items()))
                self.update({element.tag: aDict})
            # this assumes that if you've got an attribute in a tag,
            # you won't be having any text. This may or may not be a
            # good idea -- time will tell. It works for the way we are
            # currently doing XML configuration files...
            elif element.items():
                self.update({element.tag: dict(element.items())})
            # finally, if there are no child tags and no attributes, extract
            # the text
            else:
                self.update({element.tag: element.text})


def process_bounding_box(bb_fp):
    root = ElementTree.parse(bb_fp).getroot()
    dd = XmlDictConfig(root)
    height = int(dd['size']['height'])
    width = int(dd['size']['width'])

    horizontal_min = int(dd['object']['bndbox']['xmin'])
    horizontal_max = int(dd['object']['bndbox']['xmax'])

    vertical_min = int(dd['object']['bndbox']['ymin'])
    vertical_max = int(dd['object']['bndbox']['ymax'])

    return height, width, \
        vertical_min, vertical_max, horizontal_min, horizontal_max


if __name__ == '__main__':
    time_range = (0, 60)
    fragment_unit = 5  # second
    num_fragments = (time_range[1]-time_range[0]) // fragment_unit

    phase = 'te'

    # Settings
    sr = 16000
    hop = 512
    num_tags = 9

    num_frames_per_seg = 16
    target_size = None  # (height, width)

    model_id_i = '20170319_234210'
    # model_id_i = '20170319_085641'

    # model_id_o = '20170319_011751'

    param_type = 'best_measure'
    # param_type = 'best_loss'

    # Dirs and fps
    base_dir = "/home/ciaua/NAS/home/data/youtube8m/"
    base_img_dir = os.path.join(
        "/home/ciaua/NAS/home/data/imagenet/instrument.youtube8m/",
        "images")
    inst_list = sorted(os.listdir(base_img_dir))

    # Output
    out_dir = os.path.join(
        base_dir,
        'test_result.test_object_model_with_imagenet',
        model_id_i, param_type)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_fp = os.path.join(out_dir, 'result.csv')

    # Extract images
    # Tag
    # tag_fp = os.path.join(base_dir, 'tag_list.instrument.csv')
    # tag_list = [term[0] for term in it.read_csv(tag_fp)]

    # anno_dir = os.path.join(base_dir, )

    # fn_list = os.listdir(feat_dir)

    # Dirs and fps
    save_dir = os.path.join(base_dir, 'save.video')
    model_dir_i = os.path.join(save_dir, model_id_i)
    # model_dir_o = os.path.join(save_dir, model_id_o)

    # Model: Network structure
    print('Making network...')
    input_var_i = T.tensor4('visual_input.image')
    # input_var_o = T.tensor4('visual_input.optical_flow')

    net_dict_i = make_structure_image(input_var_i, num_tags)
    # net_dict_o = make_structure_optical_flow(input_var_o, num_tags)

    network_i = net_dict_i['conv8']
    # network_o = net_dict_o['conv8']

    # Load params
    param_fp_i = os.path.join(
        save_dir, model_id_i, 'model', 'params.{}.npz'.format(param_type))
    # param_fp_o = os.path.join(
    #     save_dir, model_id_o, 'model', 'params.{}.npz'.format(param_type))
    utils.load_model(param_fp_i, network_i)
    # utils.load_model(param_fp_o, network_o)

    # Compute loss
    output_va_var_i = layers.get_output(network_i, deterministic=True)
    # output_va_var_o = layers.get_output(network_o, deterministic=True)

    # Make functions
    input_var_list = [input_var_i]
    func_pr = theano.function(input_var_list, output_va_var_i)

    out_list = [['instrument', 'Hit count', 'Total count', 'Hit ratio']]
    count_all_list = list()
    for inst in inst_list:
        img_dir = os.path.join(base_img_dir, inst)
        fn_list = os.listdir(img_dir)

        inst_id = fn_list[0].split('_')[0]

        bb_dir = os.path.join(
            '../../imagenet/instrument.youtube8m/',
            'image_url_and_bounding_box', inst, 'Annotation', inst_id)

        inst_idx = inst_list.index(inst)

        out_fn_fp = os.path.join(out_dir, 'good_fn.{}.txt'.format(inst))

        count_hit = 0
        count_all = 0
        good_fn_list = list()
        for fn in fn_list:
            id_ = os.path.splitext(fn)[0]
            bb_fp = os.path.join(bb_dir, '{}.xml'.format(id_))
            height, width, vertical_min, \
                vertical_max, horizontal_min, horizontal_max = \
                process_bounding_box(bb_fp)

            min_point = (vertical_min, horizontal_min)
            max_point = (vertical_max, horizontal_max)

            img_fp = os.path.join(img_dir, fn)
            try:
                img = Image.open(img_fp).resize((width, height))
                one_image = np.transpose(np.array(img), (2, 0, 1))
                good_fn_list.append(id_)
            except Exception as e:
                print("Error: {}. {}".format(fn, repr(e)))
                continue

            count_all += 1

            # Predict
            pred_one_i = func_pr(one_image[None, :])[0, inst_idx]

            pp = imresize(pred_one_i, (height, width))

            best = np.unravel_index(pp.argmax(), pp.shape)
            hit = (min_point <= best <= max_point)

            count_hit += hit
        hit_ratio = count_hit/float(count_all)
        count_all_list.append(count_all)
        print(inst, hit_ratio, count_all)
        out_list.append([inst, count_hit, count_all, hit_ratio])
        it.write_lines(out_fn_fp, good_fn_list)
        # raw_input(123)
    it.write_csv(out_fp, out_list)
    # print(np.mean(count_all_list))

#!/usr/bin/env python

import os
from jjtorch import utils, measure
import io_tool as it
import numpy as np

import torch
# from torchnet.dataset import ListDataset
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

gid = 2  # GPU id
torch.cuda.set_device(gid)


def get_sub_prediction_from_prediction(prediction, label_idx_list_list):
    out_list = list()
    for label_idx_list in label_idx_list_list:
        temp = np.max(prediction[:, label_idx_list], axis=1, keepdims=True)
        out_list.append(temp)

    sub_prediction = np.concatenate(out_list, axis=1)
    return sub_prediction


class Net(nn.Module):
    def __init__(self, num_labels, feat_dim, scaler_list):
        super(Net, self).__init__()

        # Basic
        self.mean_list = [
            Variable(torch.FloatTensor(
                scaler.mean_[None, :, None, None]).cuda())
            for scaler in scaler_list]
        self.std_list = [
            Variable(torch.FloatTensor(
                scaler.scale_[None, :, None, None]).cuda())
            for scaler in scaler_list]

        num_types = len(scaler_list)
        self.num_types = num_types
        self.feat_dim = feat_dim

        self.num_labels = num_labels

        # Common
        self.pool = nn.MaxPool2d((4, 1), padding=(2, 0))
        self.dropout = nn.Dropout(p=0.5)

        #
        self.conv1 = nn.Conv2d(
            feat_dim*num_types, 256*num_types,
            kernel_size=(5, 1), padding=(2, 0), groups=num_types)
        self.bn1 = nn.BatchNorm2d(256*num_types)

        self.conv2 = nn.Conv2d(
            256*num_types, 256*num_types,
            kernel_size=(5, 1), padding=(2, 0), groups=num_types)
        self.bn2 = nn.BatchNorm2d(256*num_types)

        self.fconv1 = nn.Conv2d(
            256*num_types, 512*num_labels, kernel_size=1)
        self.fbn1 = nn.BatchNorm2d(512*num_labels)
        self.fconv2 = nn.Conv2d(
            512*num_labels, 512*num_labels, kernel_size=1, groups=num_labels)
        self.fbn2 = nn.BatchNorm2d(512*num_labels)

        self.fconv = nn.Conv2d(
            512*num_labels, self.num_labels, kernel_size=1, groups=num_labels)

    def forward(self, x_list):
        # Input: x, shape=(batch_size, feat_dim, num_frames, 1)

        # Normalization
        x_list = [
            (x-mu.expand_as(x))/std.expand_as(x)
            for x, mu, std in zip(x_list, self.mean_list, self.std_list)]
        x = torch.cat(x_list, dim=1)

        # Early
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Late
        x = self.dropout(F.relu(self.fbn1(self.fconv1(x))))
        x = self.dropout(F.relu(self.fbn2(self.fconv2(x))))

        # Output, shape=(bs, NLabels, NFrames)
        x = F.sigmoid(self.fconv(x))  # shape=(bs, NLables, NFrames, 1)
        x = x.view((x.size()[:3]))  # shape=(bs, NLabels, NFrames)

        pooled_x = F.avg_pool1d(x, kernel_size=x.size()[2]).view(x.size()[:2])

        return x, pooled_x


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
    model_id = '20170711_001504'
    pool_size = 16
    clip_limit = 5000

    param_type = 'best_measure'
    # param_type = 'best_loss'

    # Options
    sr = 16000
    hop_size = 512
    # test_measure_type = 'mean_auc_y'
    test_measure_type = 'auc_y_classwise'
    # test_measure_type = 'ap_y_classwise'

    feat_dim = 128
    num_labels = 9

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
    base_dir = "/home/ciaua/NAS/home/data/TMM17_instrument_playing/AudioSet/"
    tr_base_dir = "/home/ciaua/NAS/home/data/AudioSet/"
    base_data_dir = '/home/ciaua/NAS/home/data/youtube8m'

    # (for test) Dirs
    test_base_dir = '/home/ciaua/NAS/home/data/MedleyDB/'

    out_dir = os.path.join(base_dir, 'test_result.audio.frame.merge',
                           model_id)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Main
    print(model_id)

    # output
    out_fp = os.path.join(out_dir,
                          '{}.{}.csv'.format(test_measure_type, param_type))

    # model related
    save_dir = os.path.join(base_dir, 'save.audio')
    info_fp = os.path.join(save_dir, model_id, 'info.pkl')
    save_info = it.unpickle(info_fp)
    info = dict(save_info)

    # Default setting
    feat_type_list = [
        "logmelspec10000.16000_512_512_128.0.raw",
        "logmelspec10000.16000_2048_512_128.0.raw",
        "logmelspec10000.16000_8192_512_128.0.raw"
    ]
    # n_top = info['num_tags']
    num_sources = len(feat_type_list)

    # Test data
    feat_dir_list = [os.path.join(test_base_dir, 'feature', feat_type)
                     for feat_type in feat_type_list]
    anno_dir = os.path.join(test_base_dir,
                            'annotation.medleydb_yt8m',
                            '{}_{}'.format(sr, hop_size))

    # Scaler dir
    scaler_dir_list = [
        os.path.join(tr_base_dir,
                     'exp_data_common.audio.target_time.{}'.format(clip_limit),
                     ft) for ft in feat_type_list]
    scaler_fp_list = [os.path.join(scaler_dir, 'scaler.pkl')
                      for scaler_dir in scaler_dir_list]
    scaler_list = [it.unpickle(scaler_fp)for scaler_fp in scaler_fp_list]

    # Network structure
    scaler = scaler_list[0]
    net = Net(num_labels, feat_dim, scaler_list)
    net.cuda()

    # Load params
    # param_type = 'params.best_measure'
    param_fp = os.path.join(
        save_dir, model_id, 'model', 'params.{}.torch'.format(param_type))
    utils.load_model(param_fp, net)

    # file name list
    fn_list = os.listdir(anno_dir)

    # Load tag list
    model_tag_fp = os.path.join(base_data_dir, 'tag_list.instrument.csv')
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

    # Predict
    anno_all = None
    pred_all = None
    net.eval()
    for fn in sorted(fn_list):
        print(fn)
        anno_fp = os.path.join(anno_dir, fn)
        feat_fp_list = [os.path.join(feat_dir, fn)
                        for feat_dir in feat_dir_list]

        # Process annotation
        anno_ = np.load(anno_fp)
        anno = merge_anno(anno_, model_tag_list, test_tag_list, c_dict)

        n_frames = anno.shape[0]

        feats = [np.load(feat_fp) for feat_fp in feat_fp_list]
        feats = [
            Variable(torch.FloatTensor(feat.T[None, :, :, None])).cuda()
            for feat in feats]

        # Predict and upscale
        # feat = feat.cuda()
        x, pooled_x = net(feats)
        # prediction = prediction[0, :, :, 0].T
        prediction_ = x[0, :, :].data.cpu().numpy().T

        # upscale
        prediction_ = np.repeat(prediction_, pool_size, axis=0)

        prediction = prediction_[:n_frames]
        # raw_input(123)

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

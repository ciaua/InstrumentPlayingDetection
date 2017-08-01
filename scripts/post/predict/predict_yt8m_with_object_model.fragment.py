import os
import numpy as np
import io_tool as it
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

gid = 0  # GPU id
torch.cuda.set_device(gid)


def do_one(args):
    fn, base_feat_dir, base_out_dir, net, fragment_unit = args
    print(fn)

    # Get youtube id
    feat_dir = os.path.join(base_feat_dir, fn)

    if not os.path.exists(feat_dir):
        print('No feat: {}'.format(fn))
        return
    subfn_list = os.listdir(feat_dir)

    out_dir = os.path.join(base_out_dir, fn)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for subfn in subfn_list:
        feat_fp = os.path.join(feat_dir, subfn)
        out_fp = os.path.join(out_dir, subfn)

        if os.path.exists(out_fp):
            print('Done before: {}'.format(subfn))
            return

        # Extract feature
        try:
            feat = np.load(feat_fp).astype('float32')
            # print(feat.shape)
            # raw_input(123)
            # .T[None, :, :, None].astype('float32')
            feat = Variable(torch.FloatTensor(feat)).cuda()
            # print(feat_list[0].size())
            # raw_input(123)
            x, x_pooled = net(feat)
            pred = x.data.cpu().numpy()
            # print(pred.shape)
            # raw_input(123)
            # print(pred.shape)
            # raw_input(123)
            np.save(out_fp, pred)

        except Exception as e:
            print('Exception in extracting feature: {} {}. {}'.format(
                fn, subfn, repr(e)))
            return
    print('Done: {}'.format(fn))


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
    num_cores = 10

    model_id = '20170710_211725'
    param_type = 'best_measure'

    sr = 16000
    hop = 512
    num_labels = 9
    out_type = 'instrument_prediction'

    feat_type = 'image.16000_512.16_frames_per_seg.h_w_max_256'

    time_range = (0, 60)
    fragment_unit = 5  # second

    base_data_dir = "/home/ciaua/NAS/home/data/youtube8m/"
    base_dir = '/home/ciaua/NAS/home/data/TMM17_instrument_playing/YT8M/'
    save_dir = os.path.join(base_dir, 'save.image')
    model_dir = os.path.join(save_dir, model_id)

    fold_dir = '/home/ciaua/NAS/home/data/youtube8m/fold.tr16804_va2100_te2100'
    fn_fp_list = [os.path.join(fold_dir, fn) for fn in os.listdir(fold_dir)]

    base_feat_dir = os.path.join(
        base_data_dir, 'feature.{}s_fragment'.format(fragment_unit),
        'video.time_{}_to_{}'.format(*time_range),
        feat_type)

    base_out_dir = os.path.join(
        base_data_dir, 'feature.{}s_fragment'.format(fragment_unit),
        'video.time_{}_to_{}'.format(*time_range),
        '{}.{}_{}.{}'.format(out_type, sr, hop, model_id))

    # Load the scaler
    pretrained_model_dir = os.path.join(base_data_dir, 'pretrained_models')
    pretrained_model_fp = os.path.join(pretrained_model_dir,
                                       'FCN.VGG_CNN_M_2048.RGB.pytorch')
    pmodel = torch.load(pretrained_model_fp)
    mean_image = pmodel['mean_image'].mean(axis=0).mean(axis=0)

    # make net
    net = Net(num_labels, feat_dim=3, feat_mean=mean_image)
    net.eval()
    net.cuda()

    # Load param
    param_fp = os.path.join(
        save_dir, model_id, 'model', 'params.{}.torch'.format(param_type))
    utils.load_model(param_fp, net)

    # Make functions
    fn_list = list()
    for fn_fp in fn_fp_list:
        fn_list += ['{}'.format(term) for term in it.read_lines(fn_fp)]
    fn_list.sort()

    fn_list = fn_list[2::3]

    args_list = list()
    for fn in fn_list:
        args = (fn, base_feat_dir, base_out_dir, net, fragment_unit)
        args_list.append(args)
    # raw_input(123)

    # pool = Pool(processes=num_cores)
    # pool.map(do_one, args_list)
    map(do_one, args_list)

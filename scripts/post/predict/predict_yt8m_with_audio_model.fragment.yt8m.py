import os
import numpy as np
import io_tool as it
from jjtorch import utils
# from multiprocessing import Pool

import torch
# from torchnet.dataset import ListDataset
# from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

gid = 0  # GPU id
torch.cuda.set_device(gid)


def do_one(args):
    fn, base_feat_dir_list, base_out_dir, net, fragment_unit = args
    print(fn)

    # Get youtube id
    feat_dir_list = [os.path.join(base_feat_dir, fn)
                     for base_feat_dir in base_feat_dir_list]
    out_dir = os.path.join(base_out_dir, fn)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # fn_list = set.intersection(
    #     *[os.listdir(feat_dir) for feat_dir in feat_dir_list])
    subfn_list = os.listdir(feat_dir_list[0])

    for subfn in subfn_list:
        feat_fp_list = [
            os.path.join(feat_dir, subfn) for feat_dir in feat_dir_list]
        out_fp = os.path.join(out_dir, subfn)

        if os.path.exists(out_fp):
            print('Done before: {}'.format(subfn))
            return

        # Extract feature
        try:
            feat_list = [
                np.load(feat_fp).T[None, :, :, None].astype('float32')
                for feat_fp in feat_fp_list]
            feat_list = [Variable(torch.FloatTensor(feat)).cuda()
                         for feat in feat_list]
            # print(feat_list[0].size())
            # raw_input(123)
            x, x_pooled = net(feat_list)
            # print(x.size())
            pred = x.data.cpu().numpy()[0, :, :].T
            # print(pred.shape)
            # raw_input(123)
            np.save(out_fp, pred)

        except Exception as e:
            print('Exception in extracting feature: {} {}. {}'.format(
                fn, subfn, repr(e)))
            return
    print('Done: {}'.format(fn))


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
        self.pool = nn.MaxPool2d((4, 1), padding=(1, 0))
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

        self.fconv1 = nn.Conv2d(256*num_types, 512, kernel_size=1)
        self.fbn1 = nn.BatchNorm2d(512)
        self.fconv2 = nn.Conv2d(512, 512, kernel_size=1)
        self.fbn2 = nn.BatchNorm2d(512)

        self.fconv = nn.Conv2d(512, self.num_labels, kernel_size=1)

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


if __name__ == '__main__':
    num_cores = 10

    model_id = '20170719_122450'  # 16 frames per seg, pool [4, 4]
    param_type = 'best_measure'

    feat_type_list = [
        "logmelspec10000.16000_512_512_128.0.raw",
        "logmelspec10000.16000_2048_512_128.0.raw",
        "logmelspec10000.16000_8192_512_128.0.raw",
    ]
    sr = 16000
    hop = 512
    num_labels = 9
    feat_dim = 128
    out_type = 'audio_prediction'

    time_range = (0, 60)
    fragment_unit = 5  # second

    audio_data_dir = '/home/ciaua/NAS/home/data/youtube8m/'
    base_dir = '/home/ciaua/NAS/home/data/TMM17_instrument_playing/YT8M/'
    db2_dir = '/home/ciaua/NAS/Database2/YouTube8M/'
    save_dir = os.path.join(base_dir, 'save.audio')
    model_dir = os.path.join(save_dir, model_id)

    fold_dir = '/home/ciaua/NAS/home/data/youtube8m/fold.tr16804_va2100_te2100'
    fn_fp_list = [os.path.join(fold_dir, fn) for fn in os.listdir(fold_dir)]

    base_feat_dir_list = [
        os.path.join(
            db2_dir, 'feature.{}s_fragment'.format(fragment_unit),
            'audio.time_{}_to_{}'.format(*time_range), feat_type)
        for feat_type in feat_type_list
    ]

    base_out_dir = os.path.join(
        db2_dir, 'feature.{}s_fragment'.format(fragment_unit),
        'audio.time_{}_to_{}'.format(*time_range),
        '{}.{}_{}.{}'.format(out_type, sr, hop, model_id))

    # Load the scaler
    scaler_dir_list = [
        os.path.join(audio_data_dir,
                     'exp_data_common.audio.time_30_to_60',
                     ft) for ft in feat_type_list]
    scaler_fp_list = [os.path.join(scaler_dir, 'scaler.pkl')
                      for scaler_dir in scaler_dir_list]
    scaler_list = [it.unpickle(scaler_fp)for scaler_fp in scaler_fp_list]

    # make net
    net = Net(num_labels, feat_dim, scaler_list)
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

    # fn_list = fn_list[4::5]

    args_list = list()
    for fn in reversed(fn_list):
        args = (fn, base_feat_dir_list, base_out_dir, net, fragment_unit)
        args_list.append(args)
    # raw_input(123)

    # pool = Pool(processes=num_cores)
    # pool.map(do_one, args_list)
    map(do_one, args_list)

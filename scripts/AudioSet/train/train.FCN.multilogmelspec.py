#!/usr/bin/env python

import os
import time

from jjtorch import load_data as ld
from jjtorch import utils
from jjtorch import measure
from jjtorch.data import MultiTensorDataset

import numpy as np
import io_tool as it

import torch
# from torchnet.dataset import ListDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# from torch.nn import ModuleList

gid = 2  # GPU id
torch.cuda.set_device(gid)


def make_data_iterators(feat_type_list, prefix, batch_size):
    X_tr_list, y_tr, X_va_list, y_va = \
        ld.load_shared_tr_va(feat_type_list, prefix=prefix)
    num_tr = y_tr.shape[0]
    num_va = y_va.shape[0]

    # Reshape to (num_instances, feat_dim, num_frames, 1)
    X_tr_list = [torch.FloatTensor(X).transpose(1, 3) for X in X_tr_list]
    X_va_list = [torch.FloatTensor(X).transpose(1, 3) for X in X_va_list]
    y_tr = torch.FloatTensor(y_tr)
    y_va = torch.FloatTensor(y_va)

    # X_tr = X_tr.transpose(1, 3)
    # X_va = X_va.transpose(1, 3)

    # X_tr = X_tr[:1000]#
    # y_tr = y_tr[:1000]#
    # X_va = X_va[:1000]#
    # y_va = y_va[:1000]#

    dataset_tr = MultiTensorDataset(X_tr_list, y_tr)
    dataset_va = MultiTensorDataset(X_va_list, y_va)
    data_iterator_tr = DataLoader(
        dataset=dataset_tr, shuffle=True, batch_size=batch_size)
    # dataset=dataset_tr, shuffle=True, batch_size=batch_size, drop_last=True)
    data_iterator_va = DataLoader(
        dataset=dataset_va, shuffle=False, batch_size=1)

    return data_iterator_tr, data_iterator_va, num_tr, num_va


def validate(net, iterator_va, measure_func):
    sum_loss_va = 0
    count_all_va = 0
    output_va_list = list()
    target_va_list = list()
    net.eval()
    num_batches_va = len(iterator_va)
    for i_batch, batch in enumerate(iterator_va):
        count_all_va += 1

        feats = batch[:-1]
        target = batch[-1]

        feats = [feat.cuda() for feat in feats]
        target = target.cuda()

        feats = [Variable(feat, volatile=True) for feat in feats]
        target = Variable(target)

        x, pooled_x = net(feats)

        loss = criterion(pooled_x, target)
        loss_va = loss.data[0]

        target_va_list.append(target.data.cpu().numpy())
        output_va_list.append(pooled_x.data.cpu().numpy())

        sum_loss_va += loss_va
        if i_batch % 1000 == 0:
            print('{}/{}'.format(i_batch, num_batches_va))
        # if i_batch == 10:
        #     break
    all_output_va = np.vstack(output_va_list)
    all_target_va = np.vstack(target_va_list)
    score_va = measure_func(all_target_va, all_output_va)
    mean_loss_va = sum_loss_va / count_all_va

    return score_va, mean_loss_va


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
        self.pool = nn.MaxPool2d((4, 1), padding=(0, 0))
        self.dropout = nn.Dropout(p=0.5)

        #
        k = 8
        self.conv1 = nn.Conv2d(
            feat_dim*num_types, k*num_types,
            kernel_size=(5, 1), padding=(2, 0), groups=num_types)
        self.bn1 = nn.BatchNorm2d(k*num_types)

        self.conv2 = nn.Conv2d(
            k*num_types, k*num_types,
            kernel_size=(5, 1), padding=(2, 0), groups=num_types)
        self.bn2 = nn.BatchNorm2d(k*num_types)

        self.fconv1 = nn.Conv2d(k*num_types, 512, kernel_size=1)
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
    model_id = utils.get_current_time()
    print(model_id)

    # ### Options ###

    # Data options
    base_dir = "/home/ciaua/NAS/home/data/AudioSet/"
    base_out_dir = "/home/ciaua/NAS/home/data/TMM17_instrument_playing/AudioSet"
    feat_type_list = [
        "logmelspec10000.16000_512_512_128.0.raw",
        "logmelspec10000.16000_2048_512_128.0.raw",
        "logmelspec10000.16000_8192_512_128.0.raw",
    ]

    clip_limit = 5000
    tag_type = 'instrument'

    num_labels = 9
    feat_dim = 128

    # Training options
    measure_type = 'mean_auc'
    init_lr = 0.01
    num_epochs = 100
    save_rate = 1

    batch_size = 10

    # ### Misc ###
    feat_type_str = '__'.join(feat_type_list)

    # Dirs and fps
    save_dir = os.path.join(base_out_dir, 'save.audio')
    output_dir = os.path.join(save_dir, model_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    info_fp = os.path.join(output_dir, 'info.pkl')

    # ### Load data ###
    print("Loading data...")
    prefix = 'jy.audioset'
    iterator_tr, iterator_va, num_tr, num_va = make_data_iterators(
        feat_type_list, prefix, batch_size)
    print('tr: {}, va: {}'.format(num_tr, num_va))

    # Load the scaler
    scaler_dir_list = [
        os.path.join(base_dir,
                     'exp_data_common.audio.target_time.{}'.format(clip_limit),
                     ft) for ft in feat_type_list]
    scaler_fp_list = [os.path.join(scaler_dir, 'scaler.pkl')
                      for scaler_dir in scaler_dir_list]
    scaler_list = [it.unpickle(scaler_fp)for scaler_fp in scaler_fp_list]

    # ### Save info ###
    save_info = [
        ('model_id', model_id),
        ('init_lr', init_lr),
        ('num_epochs', num_epochs),
        ('base_dir', base_dir),
        ('feat_type', feat_type_str),
        ('num_labels', num_labels),
    ]
    utils.save_info(info_fp, save_info)

    # ### Build the model ###

    # Model
    net = Net(num_labels, feat_dim, scaler_list)

    net.cuda()
    criterion = nn.BCELoss().cuda()
    optimizer = optim.Adagrad(net.parameters(), lr=init_lr)

    # Measure function
    measure_func = getattr(measure, measure_type)

    # ### Training manager ###
    manager = utils.TrainingManager(net, optimizer, output_dir, save_rate)

    # ### Main ###
    manager.save_initial()
    record_title = [
        'Epoch', '(tr) Loss', '(va) Loss', '(va) Score',
        '(va) Best loss', '(va) Best loss epoch',
        '(va) Best score', '(va) Best score epoch',
        '(te) Score (Now)',
        '(te) Score (Best va loss)', '(te) Score (Best va score)',
    ]

    record = [record_title]

    for epoch in range(1, 1+num_epochs):
        print(model_id)
        t0 = time.time()

        # ### Training ###
        print('Training...')
        sum_loss_tr = 0
        count_all_tr = 0

        num_batches_tr = len(iterator_tr)
        tt0 = time.time()
        net.train()
        for i_batch, batch in enumerate(iterator_tr):
            count_all_tr += 1

            feats = batch[:-1]
            target = batch[-1]

            feats = [feat.cuda() for feat in feats]
            target = target.cuda()

            feats = [Variable(feat) for feat in feats]
            target = Variable(target)

            optimizer.zero_grad()

            x, pooled_x = net(feats)

            loss = criterion(pooled_x, target)
            loss.backward()
            optimizer.step()

            loss_tr = loss.data[0]

            sum_loss_tr += loss_tr

            # ### Print ###
            if i_batch % 100 == 0:
                print(''.join(
                    ['Epoch {}. Batch: {}/{}, T: {:.3f}, '.format(
                        epoch, i_batch, num_batches_tr, time.time()-tt0),
                        '(tr) Loss: {:.3f}, '.format(loss_tr)],
                ))
        mean_loss_tr = sum_loss_tr / count_all_tr

        # ### Validation ###
        print('')
        print('Validation...')
        score_va, mean_loss_va = validate(net, iterator_va, measure_func)

        # Check best
        best_va_loss, best_va_loss_epoch, _ = \
            manager.check_best_va_loss(mean_loss_va, epoch)
        best_va_score, best_va_score_epoch, _ = \
            manager.check_best_va_score(score_va, epoch)

        one_record = [
            epoch, mean_loss_tr, mean_loss_va, score_va,
            best_va_loss, best_va_loss_epoch,
            best_va_score, best_va_score_epoch,
        ]
        record.append(one_record)

        # Record and print
        manager.save_middle(epoch, record)

        t1 = time.time()

        print_list = ['{}: {:.3f}'.format(*term)
                      for term in zip(record_title, one_record)]
        print('Time: {:.3f}'.format(t1-t0))
        print('\n'.join([
            ', '.join(print_list[:1]),
            ', '.join(print_list[1:4]),
            ', '.join(print_list[4:6]),
            ', '.join(print_list[6:8]),
            ]
        ))

        print('')
    manager.save_final(record)
    print(model_id)

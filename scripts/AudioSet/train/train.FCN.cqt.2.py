#!/usr/bin/env python

import os
import time

from jjtorch import load_data as ld
from jjtorch import utils
from jjtorch import measure

import numpy as np
import io_tool as it

import torch
# from torchnet.dataset import ListDataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# from torch.nn import ModuleList

gid = 3  # GPU id
torch.cuda.set_device(gid)


def make_data_iterators(feat_type_list, prefix, batch_size):
    X_tr_list, y_tr, X_va_list, y_va = \
        ld.load_shared_tr_va(feat_type_list, prefix=prefix)
    num_tr = y_tr.shape[0]
    num_va = y_va.shape[0]

    X_tr = torch.FloatTensor(X_tr_list[0])
    X_va = torch.FloatTensor(X_va_list[0])
    y_tr = torch.FloatTensor(y_tr)
    y_va = torch.FloatTensor(y_va)

    # Reshape to (num_instances, feat_dim, num_frames, 1)
    X_tr = X_tr.transpose(1, 3)
    X_va = X_va.transpose(1, 3)

    # X_tr = X_tr[:1000]#
    # y_tr = y_tr[:1000]#
    # X_va = X_va[:1000]#
    # y_va = y_va[:1000]#

    dataset_tr = TensorDataset(y_tr, X_tr)
    dataset_va = TensorDataset(y_va, X_va)
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
    for i_batch, [target, feat] in enumerate(iterator_va):
        count_all_va += 1

        # feat = torch.transpose(feat, 1, 3)

        feat = feat.cuda()
        target = target.cuda()

        feat = Variable(feat, volatile=True)
        target = Variable(target)

        x, pooled_x = net(feat)

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
    def __init__(self, num_labels, feat_dim, feat_mean, feat_std):
        super(Net, self).__init__()

        # Basic
        self.mean = Variable(
            torch.FloatTensor(feat_mean[None, :, None, None]).cuda())
        self.std = Variable(
            torch.FloatTensor(feat_std[None, :, None, None]).cuda())

        self.num_labels = num_labels

        # Common
        self.pool = nn.MaxPool2d((2, 1), padding=(0, 0))
        self.dropout = nn.Dropout(p=0.5)

        #
        self.conv1 = nn.Conv2d(
            feat_dim, 256, kernel_size=(3, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(
            256, 256, kernel_size=(3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(
            256, 512, kernel_size=(3, 1), padding=(1, 0))
        self.bn3 = nn.BatchNorm2d(512)

        self.conv4 = nn.Conv2d(
            512, 512, kernel_size=(3, 1), padding=(1, 0))
        self.bn4 = nn.BatchNorm2d(512)

        self.fconv1 = nn.Conv2d(512, 512, kernel_size=1)
        self.fbn1 = nn.BatchNorm2d(512)
        self.fconv2 = nn.Conv2d(512, 512, kernel_size=1)
        self.fbn2 = nn.BatchNorm2d(512)

        self.fconv = nn.Conv2d(512, self.num_labels, kernel_size=1)

    def forward(self, x):
        # Input: x, shape=(batch_size, feat_dim, num_frames, 1)

        # Normalization
        x = (x-self.mean.expand_as(x))/self.std.expand_as(x)

        # Early
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

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
        # "logmelspec10000.16000_2048_512_128.0.raw",
        "cqt.16000_512_A0_24_176.0.raw"
    ]

    clip_limit = 5000
    tag_type = 'instrument'

    num_labels = 9
    feat_dim = 176

    # Training options
    measure_type = 'mean_auc'
    init_lr = 0.01
    num_epochs = 100
    save_rate = 1

    batch_size = 10

    # ### Misc ###
    feat_type = '__'.join(feat_type_list)

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
    scaler_dir = os.path.join(
        base_dir,
        'exp_data.audio.target_time.{}'.format(clip_limit),
        feat_type_list[0])
    scaler_fp = os.path.join(scaler_dir, 'scaler.pkl')
    scaler = it.unpickle(scaler_fp)

    # ### Save info ###
    save_info = [
        ('model_id', model_id),
        ('init_lr', init_lr),
        ('num_epochs', num_epochs),
        ('base_dir', base_dir),
        ('feat_type', feat_type),
        ('num_labels', num_labels),
    ]
    utils.save_info(info_fp, save_info)

    # ### Build the model ###

    # Model
    net = Net(num_labels, feat_dim, scaler.mean_, scaler.scale_)

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
        for i_batch, [target, feat] in enumerate(iterator_tr):
            count_all_tr += 1

            feat = feat.cuda()
            target = target.cuda()

            feat = Variable(feat)
            target = Variable(target)

            optimizer.zero_grad()

            x, pooled_x = net(feat)

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

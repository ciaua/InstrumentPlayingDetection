#!/usr/bin/env python

import os
import time
import numpy as np
from jjtorch import utils
from jjtorch import measure
from jjtorch import load_data as ld
from jjtorch.layers import SpatialCrossMapLRN as LRN
from jjtorch.optim import LSGD

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
# import torch.optim as optim
from torch.autograd import Variable

gid = 3  # GPU id
torch.cuda.set_device(gid)


def validate(net, X_va_list, y_va, measure_func):
    iterator_va = utils.make_iterator_minibatches_by_file_fragment_plus128(
        X_va_list, y_va, 1, shuffle=False, default_fragment_idx=3)

    sum_loss_va = 0
    count_all_va = 0
    output_va_list = list()
    target_va_list = list()
    net.eval()
    num_batches_va = len(y_va)
    for i_batch, batch_data in enumerate(iterator_va):
        count_all_va += 1

        feat = batch_data[0][0]
        target = batch_data[1]
        # feat = torch.transpose(feat, 1, 3)

        feat = feat.cuda()
        target = target.cuda()

        feat = Variable(feat, volatile=True)
        target = Variable(target)

        x, x_pooled = net(feat)

        loss = criterion(x_pooled, target)
        loss_va = loss.data[0]

        target_va_list.append(target.data.cpu().numpy())
        output_va_list.append(x_pooled.data.cpu().numpy())

        sum_loss_va += loss_va
        if i_batch % 1000 == 0:
            print('{}/{}'.format(i_batch, num_batches_va))
    all_output_va = np.vstack(output_va_list)
    all_target_va = np.vstack(target_va_list)
    score_va = measure_func(all_target_va, all_output_va)
    mean_loss_va = sum_loss_va / count_all_va

    return score_va, mean_loss_va


def test(net, X_te_list, y_te, measure_func):
    iterator_te = utils.make_iterator_minibatches_by_file_fragment_plus128(
        X_te_list, y_te, 1, shuffle=False, default_fragment_idx=3)

    count_all_te = 0
    output_te_list = list()
    target_te_list = list()
    net.eval()
    num_batches_te = len(y_te)
    for i_batch, batch_data in enumerate(iterator_te):
        count_all_te += 1

        feat = batch_data[0][0]
        target = batch_data[1]

        feat = feat.cuda()
        target = target.cuda()

        feat = Variable(feat, volatile=True)
        target = Variable(target)

        x, x_pooled = net(feat)

        target_te_list.append(target.data.cpu().numpy())
        output_te_list.append(x_pooled.data.cpu().numpy())

        if i_batch % 1000 == 0:
            print('{}/{}'.format(i_batch, num_batches_te))
    all_output_te = np.vstack(output_te_list)
    all_target_te = np.vstack(target_te_list)
    score_te = measure_func(all_target_te, all_output_te)

    return score_te


class Net(nn.Module):
    def __init__(self, num_labels, feat_dim):
        super(Net, self).__init__()

        # Basic
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
        # init.xavier_uniform(self.conv1.weight, gain=init.calculate_gain('relu'))
        # init.xavier_uniform(self.conv2.weight, gain=init.calculate_gain('relu'))
        # init.xavier_uniform(self.conv3.weight, gain=init.calculate_gain('relu'))
        # init.xavier_uniform(self.conv4.weight, gain=init.calculate_gain('relu'))
        # init.xavier_uniform(self.conv5.weight, gain=init.calculate_gain('relu'))
        # init.xavier_uniform(self.conv6.weight, gain=init.calculate_gain('relu'))
        # init.xavier_uniform(self.conv7.weight, gain=init.calculate_gain('relu'))
        init.xavier_uniform(self.conv1.weight)
        init.xavier_uniform(self.conv2.weight)
        init.xavier_uniform(self.conv3.weight)
        init.xavier_uniform(self.conv4.weight)
        init.xavier_uniform(self.conv5.weight)
        init.xavier_uniform(self.conv6.weight)
        init.xavier_uniform(self.conv7.weight)
        init.xavier_uniform(self.conv8.weight)
        init.constant(self.conv1.bias, 0.)
        init.constant(self.conv2.bias, 0.)
        init.constant(self.conv3.bias, 0.)
        init.constant(self.conv4.bias, 0.)
        init.constant(self.conv5.bias, 0.)
        init.constant(self.conv6.bias, 0.)
        init.constant(self.conv7.bias, 0.)
        init.constant(self.conv8.bias, 0.)

    def forward(self, x):
        # Input: x, shape=(batch_size, feat_dim, num_frames, 1)

        # Early
        x = self.pool(self.lrn1(F.relu((self.conv1(x)))))
        x = self.pool(F.relu(self.conv2(x)))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))

        # Late
        x = self.dropout(F.relu(self.conv6(x)))
        x = self.dropout(F.relu(self.conv7(x)))
        x = F.sigmoid(self.conv8(x))

        pooled_x = F.max_pool2d(x, kernel_size=x.size()[2:]).view(x.size()[:2])
        pooled_x = torch.clamp(pooled_x, 1e-6, 1-1e-6)

        return x, pooled_x


if __name__ == '__main__':
    model_type = 'fcn_optical_flow'
    model_id = utils.get_current_time()
    print(model_id)

    # Data options
    base_dir = "/home/ciaua/NAS/home/data/youtube8m/"
    base_out_dir = "/home/ciaua/NAS/home/data/youtube8m/"

    feat_type = 'dense_optical_flow'

    time_range = (0, 60)
    fragment_unit = 5  # second
    fill_factor = 2
    anno_feats_dir = os.path.join(
        base_dir,
        'exp_data.visual.time_{}_to_{}.{}s_fragment.16_frames_per_seg'.format(
            time_range[0], time_range[1], fragment_unit),
        'anno_feats.{}.fill_{}.plus128'.format(feat_type, fill_factor))
    """
    anno_feats_dir = os.path.join(
        base_dir,
        'exp_data.visual.time_{}_to_{}.{}s_fragment.16_frames_per_seg'.format(
            time_range[0], time_range[1], fragment_unit),
        'anno_feats.{}.plus127'.format(feat_type))
    """

    num_tags = 9
    feat_type_list = [feat_type]

    # Training options
    measure_type = 'mean_auc_y'
    lr = 0.001
    num_epochs = 50
    batch_size = 1  # not used in this setting

    num_cached = 3
    save_rate = 1

    # Dirs and fps
    save_dir = os.path.join(base_out_dir, 'save.video')
    output_dir = os.path.join(save_dir, model_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    info_fp = os.path.join(output_dir, 'info.pkl')

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

    # Model: Network structure
    print('Making network...')
    net = Net(num_tags, feat_dim=10)
    net.cuda()
    criterion = nn.BCELoss().cuda()
    optimizer = LSGD(net.parameters(), momentum=0.9, lr=lr)

    measure_func = getattr(measure, measure_type)

    # ### Training manager ###
    manager = utils.TrainingManager(net, optimizer, output_dir, save_rate)

    # ### Main ###
    manager.save_initial()
    record_title = [
        '(tr) Loss', '(va) Loss', '(va) Score',
        '(va) Best loss', '(va) Best loss epoch',
        '(va) Best score', '(va) Best score epoch',
        '(te) Score (Now)',
        '(te) Score (Best va loss)', '(te) Score (Best va score)',
    ]

    record = [record_title]

    save_info = [
        ('model_id', model_id),
        ('lr', lr),
        ('num_epochs', num_epochs),
        ('base_dir', base_dir),
        ('feat_type', feat_type),
        ('num_labels', num_tags),
    ]
    utils.save_info(info_fp, save_info)

    # Training
    for epoch in range(num_epochs):
        print(model_id)
        t0 = time.time()

        # Make tr iterator
        iterator_tr = \
            utils.make_iterator_minibatches_by_file_fragment_plus128(
                X_tr_list, y_tr, 1, shuffle=True)

        # ### Training ###
        print('Training...')
        sum_loss_tr = 0
        count_all_tr = 0

        num_batches_tr = len(y_tr)
        tt0 = time.time()
        net.train()
        for i_batch, batch_data in enumerate(iterator_tr):
            count_all_tr += 1

            # feat = torch.FloatTensor(batch_data[0][0])
            # target = torch.FloatTensor(batch_data[1])
            feat = batch_data[0][0]
            target = batch_data[1]
            # raw_input(123)

            feat = feat.cuda()
            target = target.cuda()

            feat = Variable(feat)
            target = Variable(target)

            optimizer.zero_grad()

            x, x_pooled = net(feat)
            # raw_input(123)

            loss = criterion(x_pooled, target)
            loss.backward()
            optimizer.step()

            loss_tr = loss.data[0]

            sum_loss_tr += loss_tr

            # ### Print ###
            if i_batch % 100 == 0:
                print(''.join(
                    ['Epoch {}. Batch: {}/{}, T: {:.3f}, '.format(
                        epoch, i_batch, num_batches_tr, time.time()-tt0),
                        '(tr) Loss: {:.3f}, '.format(loss_tr)]))
        mean_loss_tr = sum_loss_tr / count_all_tr

        # ### Validation ###
        print('')
        print('Validation...')

        score_va, mean_loss_va = validate(net, X_va_list, y_va, measure_func)

        # ### Testing ###
        print('')
        print('Testing...')

        score_te = test(
            net, X_te_list, y_te, measure_func)

        # Check best
        best_va_loss, best_va_loss_epoch, te_score_b_loss = \
            manager.check_best_va_loss(
                mean_loss_va, epoch, score_te)
        best_va_score, best_va_score_epoch, te_score_b_score = \
            manager.check_best_va_score(score_va, epoch, score_te)

        one_record = [
            mean_loss_tr, mean_loss_va, score_va,
            best_va_loss, best_va_loss_epoch,
            best_va_score, best_va_score_epoch,
            score_te, te_score_b_loss, te_score_b_score,
        ]
        record.append(one_record)

        # Record and print
        manager.save_middle(epoch, record)

        t1 = time.time()

        print_list = ['{}: {:.3f}'.format(*term)
                      for term in zip(record_title, one_record)]
        print('Time: {:.3f}'.format(t1-t0))
        print('\n'.join([
            ', '.join(print_list[:3]),
            ', '.join(print_list[3:5]),
            ', '.join(print_list[5:7]),
            ', '.join(print_list[7:8]),
            ', '.join(print_list[8:10]),
            ]
        ))

        print('')
    manager.save_final(record)

#!/usr/bin/env python

import os
import time
import numpy as np
# import io_tool as it

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


def validate(net, X_va_list, y_va, audio_threshold, inst_threshold,
             measure_func):
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
        inst = batch_data[0][1]
        audio = batch_data[1]

        baudio = (audio > audio_threshold).float()
        binst = (inst > inst_threshold).float()
        # feat = torch.transpose(feat, 1, 3)

        feat = feat.cuda()
        baudio = baudio.cuda()
        binst = binst.cuda()

        feat = Variable(feat, volatile=True)
        baudio = Variable(baudio, volatile=True)
        binst = Variable(binst, volatile=True)

        x, x_pooled = net(feat)
        artificial_target = binst*baudio[:, :, None, None].expand_as(binst)

        try:
            loss = criterion(x, artificial_target)
        except AssertionError:
            h0, w0 = x.size()[2:]
            h1, w1 = artificial_target.size()[2:]
            h = min(h0, h1)
            w = min(w0, w1)
            loss = criterion(x[:, :, :h, :w],
                             artificial_target[:, :, :h, :w])
        # loss = criterion(x, artificial_target)
        loss_va = loss.data[0]

        target_va_list.append(baudio.data.cpu().numpy())
        output_va_list.append(x_pooled.data.cpu().numpy())

        sum_loss_va += loss_va
        if i_batch % 1000 == 0:
            print('{}/{}'.format(i_batch, num_batches_va))
    all_output_va = np.vstack(output_va_list)
    all_target_va = np.vstack(target_va_list)
    score_va = measure_func(all_target_va, all_output_va)
    mean_loss_va = sum_loss_va / count_all_va

    return score_va, mean_loss_va


def test(net, X_te_list, y_te, audio_threshold, inst_threshold, measure_func):
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
        inst = batch_data[0][1]
        audio = batch_data[1]

        baudio = (audio > audio_threshold).float()
        binst = (inst > inst_threshold).float()
        # feat = torch.transpose(feat, 1, 3)

        feat = feat.cuda()
        baudio = baudio.cuda()
        binst = binst.cuda()

        feat = Variable(feat, volatile=True)
        baudio = Variable(baudio, volatile=True)
        binst = Variable(binst, volatile=True)

        x, x_pooled = net(feat)
        # artificial_target = binst*baudio.expand_as(binst)

        target_te_list.append(baudio.data.cpu().numpy())
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
    model_type = 'fcn_action_audio-target'
    model_id = utils.get_current_time()
    print(model_id)

    audio_threshold = 0.5
    inst_threshold = 0.5

    # Data options
    base_dir = "/home/ciaua/NAS/home/data/youtube8m/"
    base_out_dir = "/home/ciaua/NAS/home/data/TMM17_instrument_playing/YT8M/"

    feat_type = 'dense_optical_flow'

    time_range = (0, 60)
    fragment_unit = 5  # second

    base_action_model_id_and_fill = ('20170708_154229', 4)
    audio_model_id = '20170710_110230'
    inst_model_id = '20170710_211725'

    base_action_model_id, fill_factor = base_action_model_id_and_fill

    anno_feats_dir = os.path.join(
        base_dir,
        'exp_data.visual.time_{}_to_{}.{}s_fragment.16_frames_per_seg'.format(
            time_range[0], time_range[1], fragment_unit),
        'audioanno_feats_inst.audio_{}.inst_{}.{}.fill_{}.plus128'.format(
            audio_model_id, inst_model_id, feat_type, fill_factor))

    num_tags = 9

    # Training options
    measure_type = 'mean_auc_y'
    lr = 0.001
    num_epochs = 30
    batch_size = 10  # not used in this setting

    num_cached = 3
    save_rate = 1

    # Dirs and fps
    save_dir = os.path.join(base_out_dir, 'save.action')
    output_dir = os.path.join(save_dir, model_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        continue_training = False
    else:
        continue_training = True
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
    measure_func = getattr(measure, measure_type)

    # X_tr_list = [term[:100] for term in X_tr_list]
    # X_va_list = [term[:100] for term in X_va_list]
    # X_te_list = [term[:100] for term in X_te_list]
    # y_tr = y_tr[:100]
    # y_va = y_va[:100]
    # y_te = y_te[:100]
    # raw_input(123)

    # Model: Network structure
    print('Making network...')
    net = Net(num_tags, feat_dim=10)
    net.cuda()
    criterion = nn.BCELoss().cuda()
    optimizer = LSGD(net.parameters(), momentum=0.9, lr=lr)

    # Load params
    param_type = 'best_measure'
    base_param_fp = os.path.join(
        save_dir, base_action_model_id,
        'model', 'params.{}.torch'.format(param_type))
    utils.load_model(base_param_fp, net)

    # Continue training
    if continue_training is True:
        latest_epoch = utils.get_latest_epoch(output_dir)
        start_epoch = latest_epoch + 1
        print('Continue training from epoch {}...'.format(start_epoch))

        param_fp = os.path.join(
            output_dir, 'model', 'params.@{}.torch'.format(latest_epoch))
        utils.load_model(param_fp, net, optimizer=optimizer)
    else:
        start_epoch = 1

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

    save_info = [
        ('model_id', model_id),
        ('lr', lr),
        ('num_epochs', num_epochs),
        ('base_dir', base_dir),
        ('anno_feats_dir', anno_feats_dir),
        ('num_labels', num_tags),
        ('audio_threshold', audio_threshold),
        ('instrument_threshold', inst_threshold),
    ]
    utils.save_info(info_fp, save_info)

    # Training
    for epoch in range(start_epoch, 1+num_epochs):
        print(model_id)
        t0 = time.time()

        # Make tr iterator
        iterator_tr = \
            utils.make_iterator_minibatches_by_file_fragment_plus128(
                X_tr_list, y_tr, 1, shuffle=True, num_cached=num_cached)

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
            inst = batch_data[0][1]
            audio = batch_data[1]

            baudio = (audio > audio_threshold).float()
            binst = (inst > inst_threshold).float()
            # print(target.max(), target.mean())
            # raw_input(123)

            feat = feat.cuda()
            baudio = baudio.cuda()
            binst = binst.cuda()

            feat = Variable(feat)
            baudio = Variable(baudio)
            binst = Variable(binst)

            optimizer.zero_grad()

            x, x_pooled = net(feat)
            # raw_input(123)

            artificial_target = binst*baudio[:, :, None, None].expand_as(binst)

            try:
                loss = criterion(x, artificial_target)
            except Exception:
                h0, w0 = x.size()[2:]
                h1, w1 = artificial_target.size()[2:]
                h = min(h0, h1)
                w = min(w0, w1)
                loss = criterion(x[:, :, :h, :w],
                                 artificial_target[:, :, :h, :w])

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
        score_va, mean_loss_va = validate(
            net, X_va_list, y_va,
            audio_threshold, inst_threshold,
            measure_func)

        # ### Testing ###
        print('')
        print('Testing...')
        score_te = test(
            net, X_te_list, y_te,
            audio_threshold, inst_threshold,
            measure_func)

        # Check best
        best_va_loss, best_va_loss_epoch, te_score_b_loss = \
            manager.check_best_va_loss(
                mean_loss_va, epoch, score_te)
        best_va_score, best_va_score_epoch, te_score_b_score = \
            manager.check_best_va_score(score_va, epoch, score_te)

        one_record = [
            epoch, mean_loss_tr, mean_loss_va, score_va,
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
            ', '.join(print_list[:1]),
            ', '.join(print_list[1:4]),
            ', '.join(print_list[4:6]),
            ', '.join(print_list[6:8]),
            ', '.join(print_list[8:9]),
            ', '.join(print_list[9:11]),
            ]
        ))

        print('')
    manager.save_final(record)
    print(model_id)

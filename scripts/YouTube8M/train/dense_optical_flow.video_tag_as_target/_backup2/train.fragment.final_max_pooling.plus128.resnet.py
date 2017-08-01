#!/usr/bin/env python

import os
import time
import math
import numpy as np
from jjtorch import utils
from jjtorch import measure
from jjtorch import load_data as ld
# from jjtorch.layers import SpatialCrossMapLRN as LRN
from jjtorch.optim import LSGD

import torch
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
# import torch.nn.init as init
import torch.nn.functional as F
# import torch.nn.init as init
# import torch.optim as optim
from torch.autograd import Variable

gid = 0  # GPU id
torch.cuda.set_device(gid)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


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

        x_pooled = net(feat)

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

        x_pooled = net(feat)

        target_te_list.append(target.data.cpu().numpy())
        output_te_list.append(x_pooled.data.cpu().numpy())

        if i_batch % 1000 == 0:
            print('{}/{}'.format(i_batch, num_batches_te))
    all_output_te = np.vstack(output_te_list)
    all_target_te = np.vstack(target_te_list)
    score_te = measure_func(all_target_te, all_output_te)

    return score_te


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResFCN(nn.Module):
    def __init__(self, feat_dim, block, layers, num_classes=1000):

        #
        self.inplanes = 64
        super(ResFCN, self).__init__()
        self.conv1 = nn.Conv2d(feat_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fconv = nn.Conv2d(512*block.expansion, num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # init.xavier_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # standardization

        #
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x_pooled = F.avg_pool2d(x, x.size()[2:])
        x = F.avg_pool2d(x, x.size()[2:])

        x = torch.sigmoid(self.fconv(x))
        x = x.view(x.size()[:2])
        # x_pooled = F.max_pool2d(x, x.size()[2:])
        # x_pooled = x_pooled.view(x_pooled.size()[:2])

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


if __name__ == '__main__':
    model_type = 'fcn_optical_flow'
    model_id = utils.get_current_time()
    print(model_id)

    # Data options
    base_dir = "/home/ciaua/NAS/home/data/youtube8m/"
    base_out_dir = "/home/ciaua/NAS/home/data/TMM17_instrument_playing/YT8M/"

    feat_type = 'dense_optical_flow'

    time_range = (0, 60)
    fragment_unit = 5  # second
    fill_factor = 2
    anno_feats_dir = os.path.join(
        base_dir,
        'exp_data.visual.time_{}_to_{}.{}s_fragment.16_frames_per_seg'.format(
            time_range[0], time_range[1], fragment_unit),
        'anno_feats.{}.fill_{}.plus128'.format(feat_type, fill_factor))

    num_tags = 9
    feat_type_list = [feat_type]

    # Training options
    measure_type = 'mean_auc_y'
    lr = 0.0001
    num_epochs = 100
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
    # net = ResFCN(10, BasicBlock, [2, 2, 2, 2], num_classes=num_tags)  # 18
    net = ResFCN(10, BasicBlock, [3, 4, 6, 3], num_classes=num_tags)  # 50

    net.cuda()
    criterion = nn.BCELoss().cuda()
    optimizer = LSGD(net.parameters(), momentum=0.9, lr=lr)

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
    for epoch in range(1, 1+num_epochs):
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
            target = batch_data[1]
            # raw_input(123)

            feat = feat.cuda()
            target = target.cuda()

            feat = Variable(feat)
            target = Variable(target)

            optimizer.zero_grad()

            x_pooled = net(feat)
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

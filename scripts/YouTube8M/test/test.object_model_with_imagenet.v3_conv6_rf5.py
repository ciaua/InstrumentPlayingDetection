import os
import io_tool as it
import numpy as np
from PIL import Image
from xml.etree import cElementTree as ElementTree

from jjtorch import utils
from jjtorch.layers import SpatialCrossMapLRN as LRN
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable

gid = 0  # GPU id
torch.cuda.set_device(gid)


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

        self.conv6 = nn.Conv2d(512, 2048, kernel_size=5, stride=1, padding=2)
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
    # Settings
    num_tags = 9

    model_id_i = '20170717_135408'

    param_type = 'best_measure'
    # param_type = 'best_loss'

    # Dirs and fps
    base_dir = "/home/ciaua/NAS/home/data/TMM17_instrument_playing/YT8M/"
    base_data_dir = "/home/ciaua/NAS/home/data/youtube8m/"
    base_img_dir = os.path.join(
        "/home/ciaua/NAS/home/data/imagenet/instrument.youtube8m/",
        "images")
    inst_list = sorted(os.listdir(base_img_dir))
    pretrained_model_dir = os.path.join(base_data_dir, 'pretrained_models')
    pretrained_model_fp = os.path.join(pretrained_model_dir,
                                       'FCN.VGG_CNN_M_2048.RGB.pytorch')
    pmodel = torch.load(pretrained_model_fp)
    mean_image = pmodel['mean_image'].mean(axis=0).mean(axis=0)

    # Output
    out_dir = os.path.join(
        base_dir,
        'test_result.test_object_model_with_imagenet',
        model_id_i, param_type)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_fp = os.path.join(out_dir, 'result.csv')

    # Dirs and fps
    save_dir = os.path.join(base_dir, 'save.image')
    model_dir_i = os.path.join(save_dir, model_id_i)

    # Model: Network structure
    print('Making network...')
    net = Net(num_tags, feat_dim=3, feat_mean=mean_image)
    net.eval()
    net.cuda()

    # Load params
    param_fp_i = os.path.join(
        save_dir, model_id_i, 'model', 'params.{}.torch'.format(param_type))
    utils.load_model(param_fp_i, net)

    out_list = [['instrument', 'Hit count', 'Total count', 'Hit ratio']]
    count_all_list = list()
    for inst in inst_list:
        img_dir = os.path.join(base_img_dir, inst)
        fn_list = os.listdir(img_dir)

        inst_id = fn_list[0].split('_')[0]

        bb_dir = os.path.join(
            '../../../imagenet/instrument.youtube8m/',
            'image_url_and_bounding_box', inst, 'Annotation', inst_id)

        inst_idx = inst_list.index(inst)

        out_fn_fp = os.path.join(out_dir, 'good_fn.{}.txt'.format(inst))

        count_hit = 0
        count_all = 0
        good_fn_list = list()
        for fn in fn_list:
            print(fn)
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
            one_image = \
                Variable(torch.FloatTensor(one_image.astype('float32'))).cuda()
            pred_one, _ = net(one_image[None, :])
            upscale = nn.UpsamplingBilinear2d(size=(height, width))
            pred_one_up = upscale(pred_one)
            pred_one_up = pred_one_up[0, inst_idx]
            pred_one_up = pred_one_up.data.cpu().numpy()

            # pp = imresize(pred_one_i, (height, width))

            best = np.unravel_index(pred_one_up.argmax(), pred_one_up.shape)
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

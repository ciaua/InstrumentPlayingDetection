#!/usr/bin/env python

import time
import os
import sys
import csv
import numpy as np
import torch
from PIL import Image


# IO
ver = sys.version_info
if ver > (3, 0):
    import pickle as pk
    opts_write = {'encoding': 'utf-8', 'newline': ''}
    opts_read = {'encoding': 'utf-8'}
else:
    import cPickle as pk
    opts_write = {}
    opts_read = {}


def pickle(file_path, obj, protocol=2):
    """
    For python 3 compatibility, use protocol 2
    """
    if not file_path.endswith('.pkl'):
        file_path += '.pkl'
    with open(file_path, 'wb') as opdwf:
        pk.dump(obj, opdwf, protocol=protocol)


def unpickle(file_path):
    with open(file_path, 'rb') as opdrf:
        data = pk.load(opdrf)
        return data


def write_line(file_path, data):
    with open(file_path, 'w', **opts_write) as opdwf:
        opdwf.write(data)


def write_lines(file_path, data_list):
    with open(file_path, 'w', **opts_write) as opdwf:
        opdwf.writelines([str(term)+'\n' for term in data_list])


def read_lines(file_path):
    with open(file_path, 'r', **opts_read) as opdrf:
        data = [term.strip() for term in opdrf.readlines()]
        return data


def append_line(file_path, data):
    with open(file_path, 'a', **opts_write) as opdwf:
        opdwf.write(data)


def write_csv(file_path, data_list):
    with open(file_path, 'w', **opts_write) as opdwf:
        csv_writer = csv.writer(opdwf)
        csv_writer.writerows(data_list)


def read_csv(file_path):
    with open(file_path, 'r', **opts_read) as opdrf:
        csv_reader = csv.reader(opdrf)
        data = [term for term in csv_reader]
        return data


# Get current time
def get_current_time():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def split_data(a_list, num_parts):
    rg = list(range(len(a_list)))
    output = [[a_list[term] for term in rg[ii::num_parts]]
              for ii in range(num_parts)]
    return output


# decorator for _check_best_value to save best model
def decorator_for_save_best(output_dir, network, optimizer, epoch, best_str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            best_value, best_value_updated = func(*args, **kwargs)
            if best_value_updated:
                if output_dir is not None:
                    model_dir = os.path.join(output_dir, 'model')
                    params_best_fp = os.path.join(
                        model_dir, 'params.best_{}.torch'.format(best_str))
                    epoch_best_fp = os.path.join(
                        model_dir, 'epoch.best_{}.txt'.format(best_str))
                    save_best_params(
                        params_best_fp, network, optimizer, epoch, best_value,
                        best_str
                    )
                    write_line(epoch_best_fp, str(epoch))
            return best_value, best_value_updated
        return wrapper

    return decorator


def check_best_value(best_value, current_value, higher_better):
    '''
    value_order: str
        'descend' or 'high' or 'high first':
            highest the best

        'ascend' or 'low' or 'low first':
            lowest the best. It's for measuring loss.
    '''
    if higher_better:
        def comp_func(x, y): return x >= y
    else:
        def comp_func(x, y): return x <= y

    if comp_func(current_value, best_value):
        best_value = current_value
        is_best_value_updated = True
    else:
        is_best_value_updated = False
    return best_value, is_best_value_updated


# Manager
class TrainingManager(object):
    def __init__(self, network, optimizer, output_dir=None, save_rate=10,
                 score_higher_better=True):
        '''
        save_rate: int
            save every save_rate epochs

        '''
        self.network = network
        self.optimizer = optimizer
        self.output_dir = output_dir
        self.save_rate = save_rate

        self.score_higher_better = score_higher_better

        self.best_va_loss = np.inf
        self.best_va_loss_epoch = -1
        if score_higher_better:
            self.best_va_score = -np.inf
        else:
            self.best_va_score = np.inf
        self.best_va_score_epoch = -1

        self.te_score_from_best_va_loss = None
        self.te_score_from_best_va_score = None

    def save_initial(self):
        model_dir = os.path.join(self.output_dir, 'model')
        record_dir = os.path.join(self.output_dir, 'record')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(record_dir):
            os.mkdir(record_dir)

        # Put a file indicating the current stage
        status_fp = os.path.join(self.output_dir, 'status.txt')
        append_line(status_fp, '<Start>\n')

        # Save structure description
        description_fp = os.path.join(self.output_dir,
                                      'structure_description.csv')
        save_structure_description(description_fp, self.network)

    def save_middle(self, epoch, record):
        model_dir = os.path.join(self.output_dir, 'model')
        record_dir = os.path.join(self.output_dir, 'record')

        # Save in the middle
        if epoch % self.save_rate == 0:
            # Save the params at this epoch
            params_fp = os.path.join(model_dir,
                                     'params.@{}.torch'.format(epoch))
            save_params(params_fp, self.network, self.optimizer)

        # Save record
        record_fp = os.path.join(record_dir, 'record.@{}.csv'.format(epoch))
        save_record(record_fp, record)

        # Update status
        status_fp = os.path.join(self.output_dir, 'status.txt')
        append_line(status_fp, 'Epoch: {}\n'.format(epoch))

    def save_final(self, record):
        record_dir = os.path.join(self.output_dir, 'record')
        model_dir = os.path.join(self.output_dir, 'model')

        record_fp = os.path.join(record_dir, 'record.final.csv')

        save_record(record_fp, record)

        params_fp = os.path.join(model_dir, 'params.final.torch')

        save_params(params_fp, self.network, self.optimizer)

        # Update status
        status_fp = os.path.join(self.output_dir, 'status.txt')
        append_line(status_fp, '<Done>\n')

    def check_best_va_loss(self, va_loss, epoch, te_score=None):
        # Check best loss
        deco = decorator_for_save_best(
            self.output_dir, self.network, self.optimizer, epoch,
            best_str='loss')

        self.best_va_loss, is_updated = deco(check_best_value)(
            self.best_va_loss, va_loss, higher_better=False)
        if is_updated:
            self.best_va_loss_epoch = epoch
            if te_score is None:
                self.te_score_from_best_va_loss = None
            else:
                self.te_score_from_best_va_loss = te_score

        return self.best_va_loss, self.best_va_loss_epoch, \
            self.te_score_from_best_va_loss

    def check_best_va_score(self, va_score, epoch, te_score=None):
        # Check best score
        deco = decorator_for_save_best(
            self.output_dir, self.network, self.optimizer,
            epoch, best_str='measure')

        self.best_va_score, is_updated = deco(check_best_value)(
            self.best_va_score, va_score,
            higher_better=self.score_higher_better)
        if is_updated:
            self.best_va_score_epoch = epoch
            if te_score is None:
                self.te_score_from_best_va_score = None
            else:
                self.te_score_from_best_va_score = te_score

        return self.best_va_score, self.best_va_score_epoch, \
            self.te_score_from_best_va_score


class TrainingGANManager(object):
    def __init__(self, network_d, network_g, optimizer_d, optimizer_g,
                 output_dir=None, save_rate=1):
        '''
        save_rate: int
            save every save_rate epochs

        '''
        self.network_d = network_d
        self.network_g = network_g
        self.optimizer_d = optimizer_d
        self.optimizer_g = optimizer_g
        self.output_dir = output_dir
        self.save_rate = save_rate

        self.model_dir = os.path.join(output_dir, 'model')
        self.record_dir = os.path.join(output_dir, 'record')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        if not os.path.exists(self.record_dir):
            os.mkdir(self.record_dir)

    def save_initial(self):
        # Put a file indicating the current stage
        status_fp = os.path.join(self.output_dir, 'status.txt')
        append_line(status_fp, '<Start>\n')

        # Save structure description
        description_fp_d = os.path.join(self.output_dir,
                                        'structure_description.d.csv')
        description_fp_g = os.path.join(self.output_dir,
                                        'structure_description.g.csv')
        save_structure_description(description_fp_d, self.network_d)
        save_structure_description(description_fp_g, self.network_g)

    def save_middle(self, epoch, record):
        # Save in the middle
        if epoch % self.save_rate == 0:
            # Save the params at this epoch
            params_fp_d = os.path.join(self.model_dir,
                                       'params.d.@{}.torch'.format(epoch))
            params_fp_g = os.path.join(self.model_dir,
                                       'params.g.@{}.torch'.format(epoch))
            save_params(params_fp_d, self.network_d, self.optimizer_d)
            save_params(params_fp_g, self.network_g, self.optimizer_g)

        # Save record
        record_fp = os.path.join(
            self.record_dir, 'record.@{}.csv'.format(epoch))
        save_record(record_fp, record)

        # Update status
        status_fp = os.path.join(self.output_dir, 'status.txt')
        append_line(status_fp, 'Epoch: {}\n'.format(epoch))

    def save_final(self, record):
        record_fp = os.path.join(self.record_dir, 'record.final.csv')

        save_record(record_fp, record)

        params_fp_d = os.path.join(self.model_dir, 'params.d.final.torch')
        params_fp_g = os.path.join(self.model_dir, 'params.g.final.torch')

        save_params(params_fp_d, self.network_d, self.optimizer_d)
        save_params(params_fp_g, self.network_g, self.optimizer_g)

        # Update status
        status_fp = os.path.join(self.output_dir, 'status.txt')
        append_line(status_fp, '<Done>\n')


# Save/load
def save_best_params(fp, network, optimizer, epoch, value, best_str):
    out = {
        'epoch': epoch,
        'state_dict.model': network.state_dict(),
        'state_dict.optimizer': optimizer.state_dict(),
        'value.{}'.format(best_str): value
    }
    torch.save(out, fp)


def save_params(fp, network, optimizer):
    out = {
        'state_dict.model': network.state_dict(),
        'state_dict.optimizer': optimizer.state_dict()
    }
    torch.save(out, fp)


def load_params(fp, device_id, all_ids=[0, 1, 2, 3, 'cpu']):
    if device_id is 'cpu':
        params = torch.load(
            fp,
            map_location=lambda storage, loc: storage)
    else:
        params = torch.load(
            fp,
            map_location={
                'cuda:{}'.format(gid): 'cuda:{}'.format(device_id)
                for gid in all_ids})
    return params


def load_model(fp, network, optimizer=None, device_id=0):
    obj = load_params(fp, device_id)
    model_state_dict = obj['state_dict.model']
    if optimizer is not None:
        optimizer_state_dict = obj['state_dict.optimizer']
        optimizer.load_state_dict(optimizer_state_dict)

    network.load_state_dict(model_state_dict)


def get_structure_description(network):
    '''

    Return
    ------
    list of layer description

    '''
    # des = str(network).split('\n  ')[1:-1]
    des = str(network).split('\n  ')[1:]
    des[-1] = des[-1].replace('\n)', '')

    return des


def save_structure_description(out_fp, network):
    des = get_structure_description(network)
    write_lines(out_fp, des)


def load_structure_description(in_fp):
    des = read_lines(in_fp)
    return des


def save_info(fp, info):
    '''
    info: list of tuples
    '''
    pickle(fp, info)


def load_info(fp):
    return unpickle(fp)


def save_record(fp, record):
    '''
    info: list of tuples
    '''
    write_csv(fp, record)


# Iterator for file paths
def _load_input_and_target_by_file(
        file_index,
        inputs_fp_list_list,
        targets_fp_list, shuffle):

    # there are several temporal fragments in it
    targets_fp = targets_fp_list[file_index]

    inputs_fp_list = [_inputs_fp_list[file_index]
                      for _inputs_fp_list in inputs_fp_list_list]

    inputs_list = [np.load(inputs_fp) for inputs_fp in inputs_fp_list]
    targets = np.load(targets_fp)
    return inputs_list, targets


def iterate_minibatches_by_file(
        input_fp_list_list, target_fp_list,
        batchsize=1, shuffle=False, num_cached=4, default_fragment_idx=0):
    '''
    multisource source
    one file is a batch, batchsize is always 1

    inputs_fp_list_list: list
        len(inputs_fp_list_list) == num_feat_types
        len(inputs_fp_list_list[i]) == the number of data

        The first list contains the list of filepath to the audio features
        each file contains an array with shape
        (num_frames, feat_dim)

        The second list contains the list of filepath to the visual features
        each file contains an array with shape
        (num_images, num_channels, height, width)

    targets_fp_list: list
        each term in the list contains a file path to the target

    * batchsize is always 1

    '''
    # Check the number of files in each directory
    for input_fp_list in input_fp_list_list:
        assert(len(input_fp_list) == len(target_fp_list))

    num_files = len(target_fp_list)
    file_indices = np.arange(num_files)

    if shuffle:
        np.random.shuffle(file_indices)

    sub_file_indices_list = split_data(file_indices, num_cached)
    sentinel = object()  # guaranteed unique reference

    import Queue
    import threading
    queue_queue = Queue.Queue(maxsize=num_cached)

    def generator_one_file(file_indices, shuffle):
        # from itertools import imap
        # print("Loading data for real...")
        for file_index in file_indices:
            try:
                input_list, target = \
                    _load_input_and_target_by_file(
                        file_index, input_fp_list_list, target_fp_list,
                        shuffle)

                # Reshape the targets and inputs_list
                target = target
                # input_list = [_input for _input in _input_list]
            except Exception as e:
                print('Error in loading file: {}. {}'.format(
                    input_fp_list_list[0][file_index], repr(e)))
                # print('Error in loading file: {}'.format(repr(e)))
                continue
            yield input_list, target

    def producer_one_file(queue, sub_file_indices, shuffle):
        for item in generator_one_file(sub_file_indices, shuffle):
            # print("Loading one file...")
            queue.put(item)
        queue.put(sentinel)

    for ii in range(num_cached):
        # queue = Queue.Queue(maxsize=2)
        queue = Queue.Queue()

        # define producer (putting items into queue)
        # start producer (in a background thread)
        sub_file_indices = sub_file_indices_list[ii]
        thread = threading.Thread(target=producer_one_file,
                                  args=(queue, sub_file_indices, shuffle))
        thread.daemon = True
        thread.start()

        queue_queue.put(queue)

    # run as consumer (read items from queue, in current thread)
    while not queue_queue.empty():
        queue = queue_queue.get()
        data = queue.get()
        if data is not sentinel:
            input_list, target = data

            yield input_list, target

            queue.task_done()

            queue_queue.put(queue)
    queue_queue.task_done()
    print("Iterate Done")


# File by file with fragment. Allow compressed optical flow with .png format
def _load_one(inputs_fp):
    if inputs_fp.endswith('.png'):
        im = Image.open(inputs_fp)
        shape = tuple(map(int, im.info['shape'].split('x')))
        inputs = np.array(im)

        inputs = inputs.reshape(shape).astype('float32')
        inputs = torch.FloatTensor(inputs)
    else:
        inputs = np.load(inputs_fp).astype('float32')
        inputs = torch.FloatTensor(inputs)
    return inputs


def _load_input_and_target_by_file_fragment(file_index,
                                            inputs_fp_list_list,
                                            targets_fp_list, shuffle,
                                            default_fragment_idx=3):

    # there are several temporal fragments in it
    targets_fps = targets_fp_list[file_index]
    if shuffle:
        fragment_idx = np.random.randint(len(targets_fps))
    else:
        fragment_idx = default_fragment_idx

    inputs_fps_list = [_inputs_fp_list[file_index]
                       for _inputs_fp_list in inputs_fp_list_list]

    inputs_list = [_load_one(inputs_fps[fragment_idx])
                   for inputs_fps in inputs_fps_list]
    targets = np.load(targets_fps[fragment_idx])
    targets = torch.FloatTensor(targets)
    return inputs_list, targets


def make_iterator_minibatches_by_file_fragment(
        inputs_fp_list_list, targets_fp_list,
        batchsize=1, shuffle=False, num_cached=4, default_fragment_idx=0):
    '''
    multisource source
    divide a exp_data into several smaller files for large data
    using multiprocessing to pre-load next files

    inputs_fp_list_list: list
        len(inputs_fp_list_list) == 2
        len(inputs_fp_list_list[i]) == the number of data

        The first list contains the list of filepath to the audio features
        each file contains an array with shape
        (num_frames, feat_dim)

        The second list contains the list of filepath to the visual features
        each file contains an array with shape
        (num_images, num_channels, height, width)

    targets_fp_list: list
        each term in the list contains a file path to the target

    ** This is used for audio-visual experiments with MSD
    '''
    # Check the number of files in each directory
    for inputs_fp_list in inputs_fp_list_list:
        assert(len(inputs_fp_list) == len(targets_fp_list))

    num_files = len(targets_fp_list)
    file_indices = np.arange(num_files)

    if shuffle:
        np.random.shuffle(file_indices)

    sub_file_indices_list = split_data(file_indices, num_cached)
    sentinel = object()  # guaranteed unique reference

    import Queue
    import threading
    queue_queue = Queue.Queue(maxsize=num_cached)

    def generator_one_file(file_indices, shuffle):
        # from itertools import imap
        # print("Loading data for real...")
        for file_index in file_indices:
            try:
                _inputs_list, targets = \
                    _load_input_and_target_by_file_fragment(
                        file_index, inputs_fp_list_list, targets_fp_list,
                        shuffle, default_fragment_idx)
                # Reshape the targets and inputs_list
                inputs_list = [_inputs for _inputs in _inputs_list]
            except Exception as e:
                print('Error in loading file: {}. {}'.format(
                    inputs_fp_list_list[0][file_index], repr(e)))
                # print('Error in loading file: {}'.format(repr(e)))
                continue
            yield inputs_list, targets

    def producer_one_file(queue, sub_file_indices, shuffle):
        for item in generator_one_file(sub_file_indices, shuffle):
            # print("Loading one file...")
            queue.put(item)
        queue.put(sentinel)

    for ii in range(num_cached):
        # queue = Queue.Queue(maxsize=2)
        queue = Queue.Queue()

        # define producer (putting items into queue)
        # start producer (in a background thread)
        sub_file_indices = sub_file_indices_list[ii]
        thread = threading.Thread(target=producer_one_file,
                                  args=(queue, sub_file_indices, shuffle))
        thread.daemon = True
        thread.start()

        queue_queue.put(queue)

    # run as consumer (read items from queue, in current thread)
    while not queue_queue.empty():
        queue = queue_queue.get()
        data = queue.get()
        if data is not sentinel:
            inputs_list, targets = data

            yield inputs_list, targets

            queue.task_done()

            queue_queue.put(queue)
    print("Iterate Done")


# Plus 128 previously for dense optical flows (so have to minus 128)
def _load_one_plus128(inputs_fp):
    if inputs_fp.endswith('.png'):
        im = Image.open(inputs_fp)
        shape = tuple(map(int, im.info['shape'].split('x')))
        inputs = np.array(im)

        inputs = inputs.reshape(shape)
        inputs = (inputs - 128.).astype('float32')
        inputs = torch.FloatTensor(inputs)
    else:
        inputs = np.load(inputs_fp).astype('float32')
        inputs = torch.FloatTensor(inputs)
    return inputs


def _load_one_plus128_npy(inputs_fp):
    if inputs_fp.endswith('.png'):
        im = Image.open(inputs_fp)
        shape = tuple(map(int, im.info['shape'].split('x')))
        inputs = np.array(im)

        inputs = inputs.reshape(shape).astype('float32')
        inputs = (inputs - 128.)
    else:
        inputs = np.load(inputs_fp).astype('float32')
    return inputs


def _load_input_and_target_by_file_fragment_plus128(
        file_index,
        inputs_fps_list_list,
        targets_fps_list, shuffle,
        default_fragment_idx=3):

    # there are several temporal fragments in it
    targets_fps = targets_fps_list[file_index]
    if shuffle:
        fragment_idx = np.random.randint(len(targets_fps))
    else:
        fragment_idx = default_fragment_idx
    # print(fragment_idx)

    inputs_fps_list = [_inputs_fps_list[file_index]
                       for _inputs_fps_list in inputs_fps_list_list]

    inputs_list = [_load_one_plus128(inputs_fps[fragment_idx])
                   for inputs_fps in inputs_fps_list]
    targets = np.load(targets_fps[fragment_idx])
    targets = torch.FloatTensor(targets)
    return inputs_list, targets


def make_iterator_minibatches_by_file_fragment_plus128(
        inputs_fps_list_list, targets_fps_list,
        batchsize=1, shuffle=False, num_cached=4, default_fragment_idx=0):
    '''
    multisource source
    divide a exp_data into several smaller files for large data
    using multiprocessing to pre-load next files

    inputs_fps_list_list: list
        len(inputs_fp_list_list) == 2
        len(inputs_fp_list_list[i]) == the number of data

        The first list contains the list of filepath to the audio features
        each file contains an array with shape
        (num_frames, feat_dim)

        The second list contains the list of filepath to the visual features
        each file contains an array with shape
        (num_images, num_channels, height, width)

    targets_fps_list: list of list
        each term in the list contains a list of file paths to the target

    ** This is used for audio-visual experiments with Youtube8m
    '''
    # Check the number of files in each directory
    for inputs_fps_list in inputs_fps_list_list:
        assert(len(inputs_fps_list) == len(targets_fps_list))

    num_files = len(targets_fps_list)
    file_indices = np.arange(num_files)

    if shuffle:
        np.random.shuffle(file_indices)

    sub_file_indices_list = split_data(file_indices, num_cached)
    sentinel = object()  # guaranteed unique reference

    import Queue
    import threading
    queue_queue = Queue.Queue(maxsize=num_cached)

    def generator_one_file(file_indices, shuffle):
        # from itertools import imap
        # print("Loading data for real...")
        for file_index in file_indices:
            try:
                _inputs_list, targets = \
                    _load_input_and_target_by_file_fragment_plus128(
                        file_index, inputs_fps_list_list, targets_fps_list,
                        shuffle, default_fragment_idx)
                # Reshape the targets and inputs_list
                inputs_list = [_inputs for _inputs in _inputs_list]
            except Exception as e:
                print('Error in loading file: {}. {}'.format(
                    inputs_fps_list_list[0][file_index], repr(e)))
                # print('Error in loading file: {}'.format(repr(e)))
                continue
            yield inputs_list, targets

    def producer_one_file(queue, sub_file_indices, shuffle):
        for item in generator_one_file(sub_file_indices, shuffle):
            # print("Loading one file...")
            queue.put(item)
        queue.put(sentinel)

    for ii in range(num_cached):
        # queue = Queue.Queue(maxsize=2)
        queue = Queue.Queue()

        # define producer (putting items into queue)
        # start producer (in a background thread)
        sub_file_indices = sub_file_indices_list[ii]
        thread = threading.Thread(target=producer_one_file,
                                  args=(queue, sub_file_indices, shuffle))
        thread.daemon = True
        thread.start()

        queue_queue.put(queue)

    # run as consumer (read items from queue, in current thread)
    while not queue_queue.empty():
        queue = queue_queue.get()
        data = queue.get()
        if data is not sentinel:
            inputs_list, targets = data

            yield inputs_list, targets

            queue.task_done()

            queue_queue.put(queue)
    queue_queue.task_done()
    print("Iterate Done")


# Continue training
def get_latest_epoch(base_model_dir):
    model_dir = os.path.join(base_model_dir, 'model')
    epochs = [
        int(fn.replace('params.@', '').replace('.torch', ''))
        for fn in os.listdir(model_dir) if fn.startswith('params.@')]
    latest_epoch = max(epochs)
    return latest_epoch

import os
# import numpy as np
import io_tool as it
from sklearn.model_selection import train_test_split

video_limit_dict = {
    'tr': 2000,
    'va': 500}

base_dir = '/home/ciaua/NAS/home/data/youtube8m/'

anno_tr_dir = os.path.join(
    base_dir, 'annotation', 'train.{}'.format(video_limit_dict['tr']))

anno_vate_dir = os.path.join(
    base_dir, 'annotation', 'validate.{}'.format(video_limit_dict['va']))

fn_tr_list = os.listdir(anno_tr_dir)
fn_vate_list = os.listdir(anno_vate_dir)

ratio_va_in_va_te = 0.5
# ratio_te_in_va_te = 1-ratio_va_in_va_te

fn_va_list, fn_te_list = train_test_split(
    fn_vate_list, train_size=ratio_va_in_va_te)

'''
# count
print('va')
count = np.zeros(9)
for fn in fn_va_list:
    in_fp = os.path.join(anno_vate_dir, fn)
    count += np.load(in_fp)
print(count)

print('te')
count = np.zeros(9)
for fn in fn_te_list:
    in_fp = os.path.join(anno_vate_dir, fn)
    count += np.load(in_fp)
print(count)
'''

num_tr = len(fn_tr_list)
num_va = len(fn_va_list)
num_te = len(fn_te_list)
print(num_tr)
print(num_va)
print(num_te)

out_dir = os.path.join(base_dir,
                       'fold.tr{}_va{}_te{}'.format(num_tr, num_va, num_te))

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

out_tr_fp = os.path.join(out_dir, 'fold.tr.txt')
out_va_fp = os.path.join(out_dir, 'fold.va.txt')
out_te_fp = os.path.join(out_dir, 'fold.te.txt')

fn_tr_list = sorted([fn.replace('.npy', '') for fn in fn_tr_list])
fn_va_list = sorted([fn.replace('.npy', '') for fn in fn_va_list])
fn_te_list = sorted([fn.replace('.npy', '') for fn in fn_te_list])

it.write_lines(out_tr_fp, fn_tr_list)
it.write_lines(out_va_fp, fn_va_list)
it.write_lines(out_te_fp, fn_te_list)

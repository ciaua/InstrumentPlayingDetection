import os
import numpy as np
from jjtorch import share_memory as sm


# Share memory
def all_exist(feat_type_list, prefix='jy'):
    """
    check existence in memory
    """
    tmp_list = list()
    for phase in ['tr', 'va', 'te']:
        name_list = ['{}.X_{}.{}'.format(prefix, phase, feat_type)
                     for feat_type in feat_type_list]
        # For Python 3
        name_list = [name.encode('utf-8') for name in name_list]
        tmp_list += [sm.array_in_list(name) for name in name_list]
    for phase in ['tr', 'va', 'te']:
        name = '{}.y_{}'.format(prefix, phase)
        # For Python 3
        name = name.encode('utf-8')
        tmp_list += [sm.array_in_list(name)]
    return all(tmp_list)


def load_shared(feat_type_list, prefix='jy'):
    X_tr_list = [sm.get_array('{}.X_tr.{}'.format(prefix, feat_type))
                 for feat_type in feat_type_list]
    X_te_list = [sm.get_array('{}.X_te.{}'.format(prefix, feat_type))
                 for feat_type in feat_type_list]
    X_va_list = [sm.get_array('{}.X_va.{}'.format(prefix, feat_type))
                 for feat_type in feat_type_list]
    y_tr = sm.get_array('{}.y_tr.{}'.format(prefix, feat_type_list[0]))
    y_te = sm.get_array('{}.y_te.{}'.format(prefix, feat_type_list[0]))
    y_va = sm.get_array('{}.y_va.{}'.format(prefix, feat_type_list[0]))
    # print('{}.y_tr.{}'.format(prefix, feat_type_list[0]))

    return X_tr_list, y_tr, X_va_list, y_va, X_te_list, y_te


def load_shared_tr_va(feat_type_list, prefix='jy'):
    X_tr_list = [sm.get_array('{}.X_tr.{}'.format(prefix, feat_type))
                 for feat_type in feat_type_list]
    X_va_list = [sm.get_array('{}.X_va.{}'.format(prefix, feat_type))
                 for feat_type in feat_type_list]
    y_tr = sm.get_array('{}.y_tr.{}'.format(prefix, feat_type_list[0]))
    y_va = sm.get_array('{}.y_va.{}'.format(prefix, feat_type_list[0]))
    # print('{}.y_tr.{}'.format(prefix, feat_type_list[0]))

    return X_tr_list, y_tr, X_va_list, y_va


# Multiple features
def load2memory(data_dir, feat_type_list, prefix='jy'):
    ftype = 'npy'
    phase_list = ['tr', 'va', 'te']
    for ii, feat_type in enumerate(feat_type_list):
        print(feat_type)
        for phase in phase_list:
            print(phase)
            feat_fp = os.path.join(
                data_dir, feat_type,
                'feat.{}.{}'.format(phase, ftype))
            target_fp = os.path.join(
                data_dir, feat_type,
                'target.{}.{}'.format(phase, ftype))

            feat_arr_name = '{}.X_{}.{}'.format(prefix, phase, feat_type)
            target_arr_name = '{}.y_{}.{}'.format(prefix, phase, feat_type)

            feat_arr_name = feat_arr_name.encode('utf-8')
            target_arr_name = target_arr_name.encode('utf-8')

            if not sm.array_in_list(feat_arr_name):
                X = np.load(feat_fp)
                sm.make_array_noreturn(X, feat_arr_name)

                print(feat_fp, X.shape)
                del X

            if not sm.array_in_list(target_arr_name):
                y = np.load(target_fp)
                sm.make_array_noreturn(y, target_arr_name)

                print(target_fp, y.shape)
                del y


def load2memory_tr_va(data_dir, feat_type_list, prefix='jy'):
    ftype = 'npy'
    phase_list = ['tr', 'va']
    for ii, feat_type in enumerate(feat_type_list):
        print(feat_type)
        for phase in phase_list:
            print(phase)
            feat_fp = os.path.join(
                data_dir, feat_type,
                'feat.{}.{}'.format(phase, ftype))
            target_fp = os.path.join(
                data_dir, feat_type,
                'target.{}.{}'.format(phase, ftype))

            feat_arr_name = '{}.X_{}.{}'.format(prefix, phase, feat_type)
            target_arr_name = '{}.y_{}.{}'.format(prefix, phase, feat_type)

            feat_arr_name = feat_arr_name.encode('utf-8')
            target_arr_name = target_arr_name.encode('utf-8')

            if not sm.array_in_list(feat_arr_name):
                X = np.load(feat_fp)
                sm.make_array_noreturn(X, feat_arr_name)

                print(feat_fp, X.shape)
                del X

            if not sm.array_in_list(target_arr_name):
                y = np.load(target_fp)
                sm.make_array_noreturn(y, target_arr_name)

                print(target_fp, y.shape)
                del y


# Load by file
def load_by_file_fragment(
        anno_feats_tr_fp, anno_feats_va_fp, anno_feats_te_fp=None):
    '''
    Load one song at a time

    anno_feats_tr_fp: str
        a json file includes a dict
        each item is of the form
        id: [
                (anno_fp_0, feat_1_fp_0, feat_1_fp_0, ...),
                (anno_fp_1, feat_1_fp_1, feat_1_fp_1, ...),
                ...
            ]

    anno_feats_va_fp:
        a json file includes a dict
        each item is of the form
        id: [
                (anno_fp_0, feat_1_fp_0, feat_1_fp_0, ...),
                (anno_fp_1, feat_1_fp_1, feat_1_fp_1, ...),
                ...
            ]

    anno_feats_te_fp:
        a json file includes a dict
        each item is of the form
        id: [
                (anno_fp_0, feat_1_fp_0, feat_1_fp_0, ...),
                (anno_fp_1, feat_1_fp_1, feat_1_fp_1, ...),
                ...
            ]

    Return
    ------
    X_tr_list: list
        List of lists of lists of paths to training fatures
        Three layers of lists
        from outer to inner list:
            feat_type => id => time fragment

        Example:
        [
            [
                [fp.feat_1.id_1.time_1, fp.id_1.time2, ...],
                [fp.feat_1.id_2.time_1, fp.id_2.time2, ...],
                ...
            ],
            [
                [fp.feat_2.id_1.time_1, fp.id_1.time2, ...],
                [fp.feat_2.id_2.time_1, fp.id_2.time2, ...],
                ...
            ],
            ...
        ]

    X_va_list: list

    X_te_list: list

    y_tr: list
        List of lists of paths to training annotations
        Example:
        [
            [fp.id_1.time_1, fp.id_1.time2, ...],
            [fp.id_2.time_1, fp.id_2.time2, ...],
            ...
        ]

    y_va: list

    y_te: list

    ** This is used for Youtube8M
    '''
    import io_tool as it

    tr_dict = it.read_json(anno_feats_tr_fp)
    va_dict = it.read_json(anno_feats_va_fp)

    # keys
    tr_ids = sorted(tr_dict.keys())
    va_ids = sorted(va_dict.keys())

    # Annotation
    anno_fp_list_tr = [[term[0] for term in tr_dict[id_]] for id_ in tr_ids]
    anno_fp_list_va = [[term[0] for term in va_dict[id_]] for id_ in va_ids]

    # Audio
    num_types = len(tr_dict[tr_ids[0]][0])-1

    X_tr_list = list()
    X_va_list = list()

    if anno_feats_te_fp is not None:
        te_dict = it.read_json(anno_feats_te_fp)
        te_ids = sorted(te_dict.keys())
        anno_fp_list_te = [[term[0] for term in te_dict[id_]] for id_ in te_ids]
        X_te_list = list()

    for ii in range(1, num_types+1):
        X_tr = [[term[ii] for term in tr_dict[id_]] for id_ in tr_ids]
        X_va = [[term[ii] for term in va_dict[id_]] for id_ in va_ids]

        X_tr_list.append(X_tr)
        X_va_list.append(X_va)

        if anno_feats_te_fp is not None:
            X_te = [[term[ii] for term in te_dict[id_]] for id_ in te_ids]
            X_te_list.append(X_te)

    y_tr = anno_fp_list_tr
    y_va = anno_fp_list_va
    if anno_feats_te_fp is not None:
        y_te = anno_fp_list_te
    else:
        X_te_list = None
        y_te = None

    return X_tr_list, y_tr, X_va_list, y_va, X_te_list, y_te

import os
import numpy as np
import io_tool as it

if __name__ == '__main__':
    time_range = (30, 60)
    num_frames = 938

    feat_dim = 128
    num_labels = 9

    # Base dirs
    base_dir = '/home/ciaua/NAS/home/data/youtube8m/'

    # Data fold
    fold_dir = os.path.join(base_dir, 'fold.tr16804_va2100_te2100')

    # Annotation
    base_anno_dir = os.path.join(base_dir, 'annotation')

    # Feature
    # feat_type = 'logmelspec10000.16000_512_512_128.0.raw'
    # feat_type = 'logmelspec10000.16000_2048_512_128.0.raw'
    feat_type = 'logmelspec10000.16000_8192_512_128.0.raw'
    # feat_type = 'cqt.16000_512_A0_24_176.0.raw'

    all_feat_type_list = [
        'logmelspec10000.16000_512_512_128.0.raw',
        'logmelspec10000.16000_2048_512_128.0.raw',
        'logmelspec10000.16000_8192_512_128.0.raw'
    ]

    base_feat_dir = os.path.join(
        base_dir, 'feature', 'audio.time_{}_to_{}'.format(*time_range))
    in_feat_dir = os.path.join(base_feat_dir, feat_type)

    # common fns
    all_feat_dir_list = [os.path.join(base_feat_dir, ft)
                         for ft in all_feat_type_list]
    common_fn_set = set.intersection(
        *[set([term.replace('.npy', '') for term in os.listdir(fd)])
          for fd in all_feat_dir_list])

    # Output dirs and fps
    out_dir = os.path.join(
        base_dir,
        'exp_data_common.audio.time_{}_to_{}'.format(*time_range),
        feat_type
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    in_anno_dir_dict = {
        'tr': os.path.join(base_anno_dir, 'train.2000'),
        'va': os.path.join(base_anno_dir, 'validate.500'),
        'te': os.path.join(base_anno_dir, 'validate.500'),
    }

    # main
    for phase in ['te', 'va', 'tr']:
        print(phase)
        fold_fp = os.path.join(fold_dir, 'fold.{}.txt'.format(phase))
        fold = it.read_lines(fold_fp)
        fold = sorted(list(common_fn_set.intersection(fold)))

        in_anno_dir = in_anno_dir_dict[phase]

        out_feat_fp = os.path.join(out_dir, 'feat.{}.npy'.format(phase))
        out_anno_fp = os.path.join(out_dir, 'target.{}.npy'.format(phase))
        out_fn_fp = os.path.join(out_dir, 'fn.{}.txt'.format(phase))

        out_anno = np.zeros((len(fold), num_labels), dtype='float32')
        out_feat = np.zeros((len(fold), 1, num_frames, feat_dim),
                            dtype='float32')
        for ii, fn in enumerate(fold):
            in_feat_fp = os.path.join(in_feat_dir, '{}.npy'.format(fn))
            in_anno_fp = os.path.join(in_anno_dir, '{}.npy'.format(fn))

            try:
                in_anno = np.load(in_anno_fp)
                in_feat = np.load(in_feat_fp)

                nn = min(in_feat.shape[0], num_frames)

                out_feat[ii, 0, :nn] = in_feat[:nn]
                out_anno[ii, :] = in_anno
            except Exception as e:
                print('Fail loading data: {}. {}'.format(fn, repr(e)))
                continue

            print('Done: {}'.format(fn))

        np.save(out_anno_fp, out_anno)
        np.save(out_feat_fp, out_feat)
        out_fn_list = fold
        it.write_lines(out_fn_fp, out_fn_list)

import os
import io_tool as it
# import json


if __name__ == '__main__':
    num_frames_per_seg = 16
    feat_type = 'image.16000_512.{}_frames_per_seg.h_w_max_256'.format(
        num_frames_per_seg)
    out_feat_type = 'image'

    # Settings
    time_range = (0, 60)
    fragment_unit = 10  # second

    # compute the number of segments in a fragment
    sr = 16000
    hop = 512
    num_segments = int(round((fragment_unit*sr)/float(hop*num_frames_per_seg)))

    # Dirs
    base_dir = '/home/ciaua/NAS/home/data/youtube8m/'
    fold_dir = os.path.join(base_dir, 'fold.tr16804_va2100_te2100')

    # Annotations
    base_anno_dir = os.path.join(
        base_dir,
        'batch_annotation.{}'.format(num_segments))

    anno_dir_dict = {
        'tr': os.path.join(base_anno_dir, 'train.2000'),
        'va': os.path.join(base_anno_dir, 'validate.500'),
        'te': os.path.join(base_anno_dir, 'validate.500'),
    }

    # Output
    out_dir = os.path.join(
        base_dir,
        'exp_data.visual.time_{}_to_{}.{}s_fragment.{}_frames_per_seg'.format(
            time_range[0], time_range[1], fragment_unit, num_frames_per_seg),
        'anno_feats.{}'.format(out_feat_type))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Visual
    base_feat_dir = os.path.join(
        base_dir,
        'feature.{}s_fragment'.format(fragment_unit),
        'video.time_{}_to_{}'.format(*time_range),
        feat_type
    )

    args_list = list()
    for phase in ['tr', 'va', 'te']:
        anno_dir = anno_dir_dict[phase]
        fold_fp = os.path.join(fold_dir, 'fold.{}.txt'.format(phase))
        id_list = it.read_lines(fold_fp)

        # Output
        out_fp_fn = 'fp_dict.{}.json'.format(phase)
        out_fp_fp = os.path.join(out_dir, out_fp_fn)

        out_fp_dict = dict()
        for ii, _id in enumerate(id_list):
            print(ii)
            # anno fp
            anno_fp = os.path.join(anno_dir, '{}.npy'.format(_id))

            feat_dir = os.path.join(base_feat_dir, _id)
            if not os.path.exists(feat_dir):
                continue

            fn_list = os.listdir(feat_dir)

            out_fp_list = list()
            for fn in fn_list:
                feat_fp = os.path.join(feat_dir, fn)

                exist_list = map(os.path.exists, [anno_fp, feat_fp])
                all_good = all(exist_list)

                if all_good:
                    # print('Valid: {}'.format(sn))
                    out_fp_list.append([anno_fp, feat_fp])
                else:
                    print('Invalid: {}. {}'.format(_id, exist_list))
            if out_fp_list:
                out_fp_dict[_id] = sorted(out_fp_list)
        it.write_json(out_fp_fp, out_fp_dict)

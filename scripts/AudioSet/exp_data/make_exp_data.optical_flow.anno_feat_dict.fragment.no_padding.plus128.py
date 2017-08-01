import os
import io_tool as it
# import json


if __name__ == '__main__':
    num_frames_per_seg = 16

    # Settings
    label_type = 'instrument'
    fragment_unit = 5  # second
    fill_factor = 1

    # compute the number of segments in a fragment
    sr = 16000
    hop_size = 512

    num_segments = int(round(
        (fragment_unit*sr)/float(hop_size*num_frames_per_seg)
    ))

    num_flows_per_frame = 5
    feat_type = 'dense_optical_flow.16000_512.fill_{}.{}_frames_per_seg.{}_flows_per_frame.h_w_max_256.plus128.{}x{}xD2xD3_to_{}xD2*D3_png'.format(
        fill_factor,
        num_frames_per_seg, num_flows_per_frame,
        num_segments, num_flows_per_frame*2,
        num_segments*num_flows_per_frame*2
    )
    out_feat_type = 'dense_optical_flow'

    # Dirs
    base_dir = '/home/ciaua/NAS/home/data/AudioSet/'
    fold_dir = os.path.join(base_dir, 'song_list.{}'.format(label_type))

    # Annotations
    base_anno_dir = os.path.join(
        base_dir,
        'annotation.{}'.format(label_type))

    anno_dir_dict = {
        'tr': os.path.join(base_anno_dir, 'tr'),
        'va': os.path.join(base_anno_dir, 'va'),
    }
    fold_fp_dict = {
        'tr': os.path.join(fold_dir, 'unbalanced_train_segments.2000.csv'),
        'va': os.path.join(fold_dir, 'eval_segments.all.csv'),
    }

    # Output
    out_dir = os.path.join(
        base_dir,
        'exp_data.visual.target_time.{}s_fragment.{}_frames_per_seg'.format(
            fragment_unit, num_frames_per_seg),
        'anno_feats.{}.fill_{}.plus128'.format(out_feat_type, fill_factor))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Visual
    base_feat_dir = os.path.join(
        base_dir,
        'feature.{}s_fragment'.format(fragment_unit),
        'video.target_time',
        feat_type
    )

    args_list = list()
    for phase in ['tr', 'va']:
        anno_dir = anno_dir_dict[phase]
        fold_fp = fold_fp_dict[phase]
        id_list = [term[0] for term in it.read_csv(fold_fp)]

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

            fn_list = os.listdir(feat_dir)  # ext: .png

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

import os
import io_tool as it
# import json


if __name__ == '__main__':
    num_frames_per_seg = 16

    # Settings
    time_range = (0, 60)
    fragment_unit = 5  # second
    fill_factor = 16

    # compute the number of segments in a fragment
    sr = 16000
    hop_size = 512

    num_segments = int(round(
        (fragment_unit*sr)/float(hop_size*num_frames_per_seg)
    ))
    num_flows_per_frame = 5

    img_feat_type = 'image.16000_512.{}_frames_per_seg.h_w_max_256'.format(
        num_frames_per_seg)

    dof_feat_type = 'dense_optical_flow.16000_512.fill_{}.{}_frames_per_seg.{}_flows_per_frame.h_w_max_256.plus128.{}x{}xD2xD3_to_{}xD2*D3_png'.format(
        fill_factor,
        num_frames_per_seg, num_flows_per_frame,
        num_segments, num_flows_per_frame*2,
        num_segments*num_flows_per_frame*2
    )

    out_feat_type = 'image_and_dense_optical_flow'

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
        'anno_feats.{}.plus128'.format(out_feat_type))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Visual
    base_img_feat_dir = os.path.join(
        base_dir,
        'feature.{}s_fragment'.format(fragment_unit),
        'video.time_{}_to_{}'.format(*time_range),
        img_feat_type
    )
    base_dof_feat_dir = os.path.join(
        base_dir,
        'feature.{}s_fragment'.format(fragment_unit),
        'video.time_{}_to_{}'.format(*time_range),
        dof_feat_type
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

            img_feat_dir = os.path.join(base_img_feat_dir, _id)
            dof_feat_dir = os.path.join(base_dof_feat_dir, _id)
            if not os.path.exists(img_feat_dir) or \
                    not os.path.exists(dof_feat_dir):
                continue

            fn_list = os.listdir(img_feat_dir)  # ext: .png

            out_fp_list = list()
            for fn in fn_list:
                img_feat_fp = os.path.join(img_feat_dir, fn)
                dof_feat_fp = os.path.join(dof_feat_dir,
                                           fn.replace('.npy', '.png'))

                fps = [anno_fp, img_feat_fp, dof_feat_fp]
                exist_list = map(os.path.exists, fps)
                all_good = all(exist_list)

                if all_good:
                    # print('Valid: {}'.format(sn))
                    out_fp_list.append(fps)
                else:
                    print('Invalid: {}. {}'.format(_id, exist_list))
            if out_fp_list:
                out_fp_dict[_id] = sorted(out_fp_list)
        it.write_json(out_fp_fp, out_fp_dict)

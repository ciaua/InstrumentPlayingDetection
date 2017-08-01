import io_tool as it
import os


def get_pos_list(tags, label_dict):
    return [label_dict[tag] for tag in tags if tag in label_dict]

label_type = 'instrumentsinging'

base_dir = '/home/ciaua/NAS/home/data/AudioSet/'
base_out_dir = os.path.join(base_dir, 'videos_with_instruments')

fold_fn_list = ['eval_segments.csv', 'unbalanced_train_segments.csv']
label_fp = os.path.join(base_dir, 'tag_list.{}.csv'.format(label_type))
label_dict = dict(it.read_csv(label_fp))


for fold_fn in fold_fn_list:
    out_dir = os.path.join(base_out_dir, fold_fn.replace('.csv', ''))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_dict = dict()

    fold_fp = os.path.join(base_dir, fold_fn)
    fold = [term.split(', ') for term in it.read_lines(fold_fp)[3:]]
    for info in fold:
        id_ = info[0]
        tags = info[3][1:-1].split(',')

        pos_list = get_pos_list(tags, label_dict)
        if pos_list:
            print(id_)
            for pos in pos_list:
                try:
                    out_dict[pos].append(info[:3]+pos_list)
                except Exception as e:
                    out_dict[pos] = list()
                    out_dict[pos].append(info[:3]+pos_list)
        # raw_input(123)

    for inst in out_dict:
        out_fp = os.path.join(out_dir, '{}.csv'.format(inst))
        out_list = out_dict[inst]
        it.write_csv(out_fp, out_list)

import io_tool as it
import os


label_type = 'instrument'


# clip_limit = 'all'
# fold_fn = 'eval_segments'

clip_limit = 5000
fold_fn = 'unbalanced_train_segments'

base_dir = '/home/ciaua/NAS/home/data/AudioSet/'

base_in_dir = os.path.join(base_dir, 'videos_with_instruments')
in_dir = os.path.join(base_in_dir, fold_fn)

out_dir = os.path.join(base_dir, 'song_list.{}'.format(label_type))
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

out_fp = os.path.join(out_dir,
                      '{}.{}.csv'.format(fold_fn, clip_limit))

# fold_fn_list = ['eval_segments.csv', 'unbalanced_train_segments.csv']
label_fp = os.path.join(base_dir, 'tag_list.{}.csv'.format(label_type))
label_list = [term[1] for term in it.read_csv(label_fp)]

out_list = list()

for label in label_list:
    in_fp = os.path.join(in_dir, '{}.csv'.format(label))
    data = it.read_csv(in_fp)
    if clip_limit == 'all':
        out_list += [term for term in data if term not in out_list]
    else:
        out_list += [term for term in data[:clip_limit] if term not in out_list]

it.write_csv(out_fp, out_list)

# out_list = sorted(list(set(out_list)), key=lambda x: x[0])

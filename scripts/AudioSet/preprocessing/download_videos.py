import os
import io_tool as it
from pytube import YouTube
from multiprocessing import Pool


def get_video(args):
    out_dir, yt_id, vid_ext, base_url = args

    out_fp = os.path.join(out_dir, '{}.{}'.format(yt_id, vid_ext))
    if os.path.exists(out_fp):
        print('Done before. {}'.format(yt_id))
        return

    try:
        yt_link = '{}{}'.format(base_url, yt_id)
        yt = YouTube(yt_link)
        yt.set_filename(yt_id)

        # Video
        vid_format = yt.filter(vid_ext)[0]
        resolution = vid_format.resolution
        video = yt.get(vid_ext, resolution)

        # save
        video.download(out_dir)

        print('Done. {}'.format(yt_id))
        # return yt
    except Exception as e:
        print('Error. {}. {}'.format(yt_id, repr(e)))


def startswith(subject_str, initial_str_list):

    return any([subject_str.startswith(init_str)
                for init_str in initial_str_list])


if __name__ == '__main__':
    num_cores = 20
    in_fn_list = ['unbalanced_train_segments.5000.csv',
                  'eval_segments.all.csv']

    base_url = "https://www.youtube.com/watch?v="

    base_dir = '/home/ciaua/NAS/home/data/AudioSet/'

    out_dir = '/home/ciaua/NAS/Database2/AudioSet/video/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    vid_ext = 'mp4'

    args_list = list()

    for in_fn in in_fn_list:
        id_fp = os.path.join(
            base_dir, 'song_list.instrument', in_fn)
        raw_id_list = it.read_csv(id_fp)
        id_list = [term[0] for term in raw_id_list]

        for yt_id in reversed(id_list):
            args = (out_dir, yt_id, vid_ext, base_url)
            args_list.append(args)
        # yt = get_video(args)
    raw_input(123)

    pool = Pool(processes=num_cores)
    pool.map(get_video, args_list)
    # map(get_video, args_list)

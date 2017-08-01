import os
from pytube import YouTube
from multiprocessing import Pool


def get_video(args):
    out_dir, fn, vid_ext, base_url = args

    yt_id = fn.replace('.npy', '')

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


if __name__ == '__main__':
    num_cores = 20

    base_url = "https://www.youtube.com/watch?v="

    video_limit_dict = {
        'train': 5000,
        'validate': 500}

    base_dir = '/home/ciaua/NAS/home/data/youtube8m/'
    base_anno_dir = os.path.join(base_dir, 'annotation')
    out_dir = os.path.join(base_dir, 'video')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    base_id_dir = \
        '/home/ciaua/NAS/home/data/MSD/Last.fm.separated_tags/youtube_id/'

    vid_ext = 'mp4'

    args_list = list()

    for phase in ['train', 'validate']:
        video_limit = video_limit_dict[phase]
        in_dir = os.path.join(base_anno_dir, '{}.{}'.format(phase, video_limit))
        fn_list = os.listdir(in_dir)
        for fn in reversed(fn_list):
            args = (out_dir, fn, vid_ext, base_url)
            args_list.append(args)
        # yt = get_video(args)

    pool = Pool(processes=num_cores)
    pool.map(get_video, args_list)
    # map(get_video, args_list)

import os
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


if __name__ == '__main__':
    num_cores = 5

    base_url = "https://www.youtube.com/watch?v="

    db2_dir = '/home/ciaua/NAS/Database2/YouTube8M/'
    out_dir = os.path.join(db2_dir, 'video')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    id_list = [
        'Jo5kfRtvVI8',
        'bKGk3SAD3B4'
    ]
    """
        'ViCwtWZT9uo',
        'PLEhGSQqCfQ'
        'd3J_aYbTaEE',
        'rS_ug0thcm8'
        '2G2VaBX24So'
        'kLpI_KLeEkI',
        'BGjmiOkLv8M',
        'G-FIkfZ4-oA',
        'bz_VT01Kx9A',
        'lxDdJM78qHw',
        'xsz19aDi2GE',
        'Wlm5X42Dlac',
        'MVF5F_Q6Y8k',
        'BAieBB1yhfw',
        'W67_sk4B1qE',
        'Dewv42ggvHs',
        'BD2lrlVHdhc',
        '0l9EXsA2nDk'
        'Fe7PHwqqeAE',
        'CWZmeJAE70Q',
        'RZF_H8basmQ',
        '2GNDPiZGXdE',
        '7Wh8rE9zEaA',
        '0tK-_WlreF4',
        'mvDYAXbNFv8',
        '55_RhFOyRgk',
        'rXq5WcasEdc',
        '780UJ-EaeiI',
        'MVF5F_Q6Y8k',
        'BAieBB1yhfw',
        'W67_sk4B1qE',
        'Vx9UZ_NP-EI',
        '_mpZqTZxJrU',
        'ZCydAxIV-0Y',
        '7N3jE_4t46E',
        '7Hc4hks6hSs',
        'u9fnlRF56lM',
        '3hjHJo452dY',
        'AoWlQSjjxJQ',
        'jcYBbusMfro',
    """

    vid_ext = 'mp4'

    args_list = list()

    for yt_id in id_list:
        args = (out_dir, yt_id, vid_ext, base_url)
        args_list.append(args)

    pool = Pool(processes=num_cores)
    pool.map(get_video, args_list)
    # map(get_video, args_list)

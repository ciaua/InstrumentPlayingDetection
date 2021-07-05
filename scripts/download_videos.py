# -*- coding: utf-8 -*-

# from __future__ import unicode_literals
# from shutil import copyfile

import os
import youtube_dl
# import json


def download(youtube_id, out_dir):
    '''
    https://github.com/ytdl-org/youtube-dl/blob/master/README.md#embedding-youtube-dl
    https://github.com/ytdl-org/youtube-dl/issues/5192
    '''
    url = 'https://www.youtube.com/watch?v={}'.format(youtube_id)

    out_fp = os.path.join(out_dir, f'{youtube_id}.mp4')

    if os.path.exists(out_fp):
        print('Done before')
        return

    ydl_opts = {
        'format': 'mp4',
        # 'postprocessors': [{
        #     'key': 'FFmpegExtractAudio',
        #     'preferredcodec': 'mp4',
        #     'preferredquality': '192',
        # }],
        'outtmpl': os.path.join(out_dir, '%(id)s.%(ext)s'),
        # 'outtmpl': os.path.join(output_dir, '/%(title)s-%(id)s.%(ext)s')
        'fragment_retries': 10,
        'retries': 10,
        'sleep_interval': 10,
        'max_sleep_interval': 20,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
        except youtube_dl.utils.DownloadError:
            print('Video unavailable')
            return


if __name__ == '__main__':
    out_dir = '../sample_videos/'
    os.makedirs(out_dir, exist_ok=True)

    ids = [
        'd3J_aYbTaEE',
    ]

    for youtube_id in ids:
        print(youtube_id)
        download(youtube_id, out_dir)

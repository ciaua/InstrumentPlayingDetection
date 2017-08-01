import os
import io_tool as it
import moviepy.editor as mpy
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from multiprocessing import Pool


def main(args):
    youtube_id, video_dir, out_video_dir, time_range = args

    # Get youtube id
    # video
    video_fp = os.path.join(video_dir, '{}.mp4'.format(youtube_id))

    if not os.path.exists(video_fp):
        print('No video: {}.'.format(youtube_id))
        return
    vid = mpy.VideoFileClip(video_fp)
    print(vid.duration, time_range)
    if vid.duration < time_range[1]:
        print('Too short: {}.'.format(fn))
        return

    out_video_fp = os.path.join(out_video_dir, '{}.mp4'.format(youtube_id))
    if os.path.exists(out_video_fp):
        print('Done before: {}'.format(video_fp))
        return

    # Extract subclip
    # subclip = vid.subclip(*time_range)
    # subclip.write_videofile(out_video_fp)
    ffmpeg_extract_subclip(video_fp, *time_range, targetname=out_video_fp)
    print('Done: {}'.format(video_fp))


if __name__ == '__main__':
    num_cores = 10
    vid_ext = '.mp4'

    fold_dir = '/home/ciaua/NAS/home/data/AudioSet/song_list.instrument/'

    data_dir = '/home/ciaua/NAS/home/data/AudioSet/'
    db2_dir = '/home/ciaua/NAS/Database2/AudioSet/'
    video_dir = os.path.join(db2_dir, 'video')
    out_video_dir = os.path.join(db2_dir, 'video.target_time')

    if not os.path.exists(out_video_dir):
        os.makedirs(out_video_dir)

    fn_fp_list = [os.path.join(fold_dir, fn) for fn in os.listdir(fold_dir)]
    info_list = list()
    for fn_fp in fn_fp_list:
        info_list += [(term[0], tuple(term[1:3]))
                      for term in it.read_csv(fn_fp)]

    args_list = list()
    for info in info_list:
        fn = info[0]
        time_range = tuple(map(int, map(float, info[1])))
        args = (fn, video_dir, out_video_dir, time_range)
        args_list.append(args)

    pool = Pool(processes=num_cores)
    pool.map(main, args_list)
    pool.close()
    # map(main, args_list)

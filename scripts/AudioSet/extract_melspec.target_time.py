import os
from jjtorch import utils
import librosa
import numpy as np
import subprocess
from multiprocessing import Pool


def get_audio_from_video(video_fp, out_audio_fp, sr):
    if os.path.exists(out_audio_fp):
        print('Audio extracted before: {}'.format(video_fp))
        return
    try:
        command = \
            u"ffmpeg -y -i \"{}\" -b:a 128k -ac 1 -ar {} -vn \"{}\" -hide_banner -loglevel panic". format(
                video_fp, sr, out_audio_fp)
        subprocess.call(command, shell=True)
    except Exception as e:
        print('Exception in video2audio: {}. {}'.format(video_fp, repr(e)))


def main(args):
    youtube_id, video_dir, out_audio_dir, out_audio_feat_dir, \
        sr, win_size, hop_size, num_mels, time_range = args

    # Get youtube id
    # video
    video_fp = os.path.join(video_dir, '{}.mp4'.format(youtube_id))

    # Fps
    out_audio_fp = os.path.join(out_audio_dir, '{}.mp3'.format(youtube_id))
    out_audio_feat_fp = os.path.join(out_audio_feat_dir,
                                     '{}.npy'.format(youtube_id))
    if os.path.exists(out_audio_feat_fp):
        print('Done before: {}'.format(video_fp))
        return

    # Extract audio
    get_audio_from_video(video_fp, out_audio_fp, sr)

    # Extract feature
    try:
        duration = librosa.core.get_duration(filename=out_audio_fp)
        if duration < time_range[1]:
            print('Audio too short: {}'.format(video_fp))
            return
        sig, sr = librosa.core.load(out_audio_fp, sr=sr,
                                    offset=time_range[0],
                                    duration=time_range[1]-time_range[0])
        feat_ = librosa.feature.melspectrogram(sig, sr=sr,
                                               n_fft=win_size,
                                               hop_length=hop_size,
                                               n_mels=num_mels).T
        feat = np.log(1+10000*feat_)
        np.save(out_audio_feat_fp, feat)
        print('Done: {} -- {}'.format(video_fp, youtube_id))

    except Exception as e:
        print('Exception in extracting feature: {}. {}'.format(video_fp,
                                                               repr(e)))
        return


if __name__ == '__main__':
    num_cores = 10

    feat_type = 'logmelspec10000'
    sr = 16000

    # win_size = 8192
    # win_size = 2048
    win_size = 512

    hop_size = 512
    num_mels = 128

    base_dir = 'path/to/data/folder'
    # base_dir = '/home/ciaua/NAS/Database2/AudioSet/'
    video_dir = os.path.join(base_dir, 'video')

    fold_dir = '../../data/song_list/AudioSet/'

    out_audio_dir = os.path.join(base_dir, 'audio')
    out_audio_feat_dir = os.path.join(
        base_dir, 'feature', 'audio.target_time',
        '{}.{}_{}_{}_{}.0.raw'.format(
            feat_type,
            sr,
            win_size,
            hop_size,
            num_mels)
    )

    if not os.path.exists(out_audio_dir):
        os.makedirs(out_audio_dir)
    if not os.path.exists(out_audio_feat_dir):
        os.makedirs(out_audio_feat_dir)

    fn_fp_list = [os.path.join(fold_dir, fn) for fn in os.listdir(fold_dir)]
    info_list = list()
    for fn_fp in fn_fp_list:
        info_list += [('{}'.format(term[0]), tuple(term[1:3]))
                      for term in utils.read_csv(fn_fp)]

    args_list = list()
    for info in info_list:
        youtube_id = info[0]
        time_range = tuple(map(int, map(float, info[1])))
        args = (youtube_id, video_dir, out_audio_dir, out_audio_feat_dir,
                sr, win_size, hop_size, num_mels, time_range)
        args_list.append(args)
    # raw_input(123)

    pool = Pool(processes=num_cores)
    pool.map(main, args_list)
    pool.close()
    # map(get_video, args_list)

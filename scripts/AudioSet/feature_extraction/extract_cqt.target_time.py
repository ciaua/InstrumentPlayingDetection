import os
import io_tool as it
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
        sr, hop_size, fmin, bins_per_oct, num_bins = args

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
        cqt = librosa.cqt(
            sig, sr=sr, hop_length=hop_size, fmin=librosa.note_to_hz(fmin),
            n_bins=num_bins, bins_per_octave=bins_per_oct
        ).T
        cqt_amp = np.abs(cqt)
        np.save(out_audio_feat_fp, cqt_amp)
        print('Done: {} -- {}'.format(video_fp, youtube_id))

    except Exception as e:
        print('Exception in extracting feature: {}. {}'.format(video_fp,
                                                               repr(e)))
        return


if __name__ == '__main__':
    num_cores = 10

    feat_type = 'cqt'
    sr = 16000
    hop_size = 512
    fmin = 'A0'
    bins_per_oct = 24
    num_bins = 176

    fold_dir = '/home/ciaua/NAS/home/data/AudioSet/song_list.instrument/'
    fn_fp_list = [os.path.join(fold_dir, fn)
                  for fn in ['eval_segments.all.csv',
                             'unbalanced_train_segments.5000.csv']]

    data_dir = '/home/ciaua/NAS/home/data/AudioSet/'
    db2_dir = '/home/ciaua/NAS/Database2/AudioSet/'
    video_dir = os.path.join(db2_dir, 'video')

    out_audio_dir = os.path.join(db2_dir, 'audio')
    out_audio_feat_dir = os.path.join(
        db2_dir, 'feature', 'audio.target_time',
        '{}.{}_{}_{}_{}_{}.0.raw'.format(
            feat_type,
            sr, hop_size, fmin, bins_per_oct, num_bins))

    if not os.path.exists(out_audio_dir):
        os.makedirs(out_audio_dir)
    if not os.path.exists(out_audio_feat_dir):
        os.makedirs(out_audio_feat_dir)

    info_list = list()
    for fn_fp in fn_fp_list:
        info_list += [('{}'.format(term[0]), tuple(term[1:3]))
                      for term in it.read_csv(fn_fp)]

    args_list = list()
    for info in info_list:
        youtube_id = info[0]
        time_range = tuple(map(int, map(float, info[1])))
        args = (youtube_id, video_dir, out_audio_dir, out_audio_feat_dir,
                sr, hop_size, fmin, bins_per_oct, num_bins)
        args_list.append(args)
    # raw_input(123)

    pool = Pool(processes=num_cores)
    pool.map(main, args_list)
    pool.close()
    # map(get_video, args_list)

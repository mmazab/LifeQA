#!/usr/bin/env python
import os
from collections import Counter, defaultdict
from subprocess import CalledProcessError, check_call, DEVNULL
from typing import Iterable, Mapping, Tuple, Optional

import numpy as np
import scipy.fft
import scipy.io.wavfile
import scipy.signal
import skimage.util

import scripts.util


# Part of the code was copied from VideoSync: https://github.com/allisonnicoledeal/VideoSync


def read_audio(audio_file: str) -> Tuple[int, np.ndarray]:
    return scipy.io.wavfile.read(audio_file)


def fft_intensities(a: np.ndarray) -> np.ndarray:
    fft = scipy.fft.fft(a.astype(np.float32), overwrite_x=True, workers=-1)
    return np.absolute(fft[..., :fft.shape[-1] // 2])


def make_horiz_bins(audio: np.ndarray, fft_bin_size: int, overlap: int, box_height: int) -> np.ndarray:
    samples = skimage.util.view_as_windows(audio, fft_bin_size, step=fft_bin_size - overlap)
    intensities = fft_intensities(samples)
    assert (fft_bin_size // 2) % box_height == 0
    return intensities.reshape((intensities.shape[0], -1, box_height))


def make_vert_bins(horiz_bins: np.ndarray, box_width: int) -> np.ndarray:
    num_samples = horiz_bins.shape[0]
    return horiz_bins[:num_samples - num_samples % box_width].reshape((-1, box_width) + horiz_bins.shape[1:])


def peaks(boxes: np.ndarray, samples_per_box: int) -> Mapping[int, Iterable[int]]:
    boxes = boxes.transpose((2, 0, 1, 3))

    horiz_bin_count, vert_bin_count, box_width, box_height = boxes.shape

    boxes = boxes.reshape(boxes.shape[:2] + (-1,))

    indices_peaks = boxes.argpartition(- samples_per_box)[..., - samples_per_box:]

    # We compute the frequency and sample indices for the peaks.
    indices_freqs = np.arange(horiz_bin_count)[:, np.newaxis, np.newaxis] * box_height + indices_peaks % box_height
    indices_samples = np.arange(vert_bin_count)[np.newaxis, :, np.newaxis] * box_width + indices_peaks // box_height

    freqs_dict = defaultdict(list)
    for freq_idx, sample_idx in np.nditer([indices_freqs, indices_samples]):
        freqs_dict[freq_idx.item()].append(sample_idx.item())
    return freqs_dict


def audio_features(audio: np.ndarray, fft_bin_size: int, overlap: int, box_height: int, box_width: int,
                   samples_per_box: int) -> Mapping[int, Iterable[int]]:
    if audio.ndim == 2:
        audio = audio.mean(-1)
    bins_dict = make_horiz_bins(audio, fft_bin_size, overlap, box_height)
    boxes = make_vert_bins(bins_dict, box_width)
    return peaks(boxes, samples_per_box)


def most_likely_delay(features1: Mapping[int, Iterable[int]], features2: Mapping[int, Iterable[int]]) -> float:
    keys = set(features1.keys()) & set(features2.keys())
    return Counter(t1 - t2 for key in keys for t1 in features1[key] for t2 in features2[key]).most_common()[0][0]


def resample(audio: np.ndarray, old_rate: int, new_rate: int) -> np.ndarray:
    secs = len(audio) // old_rate
    new_sample_count = secs * new_rate
    return scipy.signal.resample(audio, new_sample_count)


def find_timestamps(audio_path1: str, audio_path2: str, fft_bin_size: int = 1024, overlap: int = 0,
                    box_height: int = 512, box_width: int = 32,
                    samples_per_box: int = 8) -> Optional[Tuple[float, float]]:
    """Finds the most likely start and end times of `audio_path2` within `audio_path1`, in seconds."""
    rate, audio1 = read_audio(audio_path1)
    if not audio1.any():
        return None
    features1 = audio_features(audio1, fft_bin_size, overlap, box_height, box_width, samples_per_box)

    rate2, audio2 = read_audio(audio_path2)
    if not audio2.any():
        return None
    duration = len(audio2) / rate2
    if rate != rate2:
        audio2 = resample(audio2, rate2, rate)
    features2 = audio_features(audio2, fft_bin_size, overlap, box_height, box_width, samples_per_box)

    delay = most_likely_delay(features1, features2)
    samples_per_sec = rate / fft_bin_size
    delay_in_seconds = delay / samples_per_sec

    if -0.1 < delay_in_seconds < 0:  # It's probably zero but there's an approximation error.
        delay_in_seconds = abs(delay_in_seconds)

    assert delay_in_seconds >= 0

    return delay_in_seconds, delay_in_seconds + duration


def extract_and_save_audio(input_path: str, output_path: str) -> None:
    check_call(["ffmpeg", "-y", "-i", input_path, "-vn", "-ac", "1", "-f", "wav", output_path], stdout=DEVNULL,
               stderr=DEVNULL)


def main():
    data_dicts_splits = scripts.util.load_data()
    data_dicts = {id_: data_dict for data_dicts in data_dicts_splits for id_, data_dict in data_dicts.items()}

    for id_ in sorted(data_dicts):
        data_dict = data_dicts[id_]

        video_audio_path = f"data/audios/{id_}.wav"
        if not os.path.isfile(video_audio_path):
            video_path = f"data/videos/{id_}.mp4"
            extract_and_save_audio(video_path, video_audio_path)

        parent_video_url = data_dict["parent_video_id"]
        parent_video_id = parent_video_url.split("=")[1]
        parent_audio_path = f"data/parent_audios/{parent_video_id}.wav"
        if not os.path.isfile(parent_audio_path):
            try:
                check_call(["youtube-dl", "-x", "--audio-format", "wav", "--audio-quality", "0",
                            "-o", f"{parent_audio_path[:-4]}.%(ext)s", parent_video_url],
                           stdout=DEVNULL, stderr=DEVNULL)
            except CalledProcessError:
                parent_audio_path = None

        if parent_audio_path:
            timestamps = find_timestamps(parent_audio_path, video_audio_path)
            if timestamps:
                start_time, end_time = timestamps
                print(f"{id_} {start_time:.2f} - {end_time:.2f}")
            else:
                print(f"{id_} (empty audio)")
        else:
            print(f"{id_} (deleted video)")


if __name__ == "__main__":
    main()

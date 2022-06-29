import json
import math
import os
import re
import warnings

import librosa
import numpy as np

warnings.filterwarnings('ignore')




# Делит аудио файл(фрагменты) на чанки
def chunkizer(chunk_length, audio, sr):
    num_chunks = math.ceil(librosa.get_duration(audio, sr=sr) / chunk_length)
    chunks = []
    for i in range(num_chunks):
        chunks.append(audio[i * chunk_length * sr:(i + 1) * chunk_length * sr])
    return chunks


# Эффекты
def stretch(data, rate=0.75):
    return librosa.effects.time_stretch(data, rate)


def pitch(data, sampling_rate, n_steps):
    return librosa.effects.pitch_shift(data, sampling_rate, n_steps=n_steps, bins_per_octave=12)

# Создает дополнительные экземляры аудио с небольшими изменеиями
def augmentation(folder_name, audio_name):
    audio, sr = librosa.load(folder_name + '/' + audio_name)
    pitch_up = pitch(audio, sr, 3)
    pitch_down = pitch(audio, sr, -2)
    stretched = stretch(audio)

    sf.write(folder_name + '/' + 'up_' + audio_name[:-4] + '.wav', pitch_up, sr)
    sf.write(folder_name + '/' + 'down_' + audio_name[:-4] + '.wav', pitch_down, sr)
    sf.write(folder_name + '/' + 'str_' + audio_name[:-4] + '.wav', stretched, sr)

# Выравнивает громкость к заданному среднему значению
def normalization(signal, target_dBFS):
    change_in_dBFS = target_dBFS - signal.dBFS
    return signal.apply_gain(change_in_dBFS)

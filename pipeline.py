import warnings

import librosa
import numpy as np
from decord import AudioReader, bridge
from src import utils
from src.AudioModel import AudioModel
from src.Torch_emotion import Torch_emotion

import torch
from torch import load

warnings.filterwarnings('ignore')

bridge.set_bridge(new_bridge="torch")


def amplitude_envelope(signal, frame_size=1024):
    amplitude_envelope = []
    for i in range(0, len(signal), frame_size):
        current = max(signal[i:i+frame_size])
        amplitude_envelope.append(current)
    return amplitude_envelope

def calculate_angles(envelope):
    # Returns 2 dictionaries with angles and timestams for increasing and decreasing
    increases = {}
    decreases = {}
    for i in range(len(envelope)-1):
        angle = np.rad2deg(np.arctan(envelope[i+1]-envelope[i]))
        if angle >= 0:
            increases.update({i+1: angle})
        else:
            decreases.update({i+1: angle})
    # To calculate the last angle
    if len(envelope) != len(increases) + len(decreases):
        try:
            angle = np.rad2deg(np.arctan(envelope[-1]-envelope[-2]))
        except IndexError:
            angle = 0
        if angle >= 0:
            increases.update({len(envelope)-1: angle})
        else:
            decreases.update({len(envelope)-1: angle})
    return increases, decreases

def get_rapidness(sequence, envelope, type='Volume'):
    sharp_angles = {}
    for timestamp, angle in sequence.items():
        if angle > 0:
            if angle >= np.mean(list(sequence.values())) + 2.5 * np.std(list(sequence.values())):
                sharp_angles.update({timestamp:angle})
        else:
            if angle <= np.mean(list(sequence.values())) - 2.5 * np.std(list(sequence.values())):
                sharp_angles.update({timestamp:angle})
    if type=='Volume':
        timestamps = []
        # Loudness detection
        for timestamp in sharp_angles.keys():
            try:
                if 1 - np.abs(envelope[timestamp]) < np.abs(envelope[timestamp]) - np.abs(np.mean(envelope)):
                    timestamps.append(timestamp)
            except IndexError:
                continue
        return timestamps
    else:
        return sharp_angles.keys()

def get_temp(signal, sr):
    onset_env = librosa.onset.onset_strength(signal[0], sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    return round(tempo[0], 1)

def get_emotions(signal, sr, device, model_path='./model/Emotion_classifier.pth'):
    Emos = {0: 'angry', 1: 'disgust', 2: 'happiness', 3: 'neutral', 4: 'fear', 5: 'surprise', 6: 'sadness'}
    model = AudioModel().to(device)
    model.load_state_dict(load(model_path, map_location=device))
    emotion = Torch_emotion(model, signal, sr=sr, device='device')
    emotions = emotion.audio_pipeline()
    emo_scores = []
    for i in range(len(Emos)):
        emo_scores.append(np.asarray(emotions)[:, i].sum())
    # Average emotion scores
    final_emo_score = [round(x / sum(emo_scores), 3) for x in emo_scores]
    
    # Change sequence of emotions 
    new_order = [0, 1, 3, 2, 4, 5, 6]

    final_emo_score = [final_emo_score[i] for i in new_order]

    return final_emo_score

def sound_markers(path, sample_rate=44100, device='cpu', timestamps=False, name='',
annotation=False, duration=10, model_path='./model/Emotion_classifier.pth'):

    sr = sample_rate

    signal = AudioReader(path, sample_rate=sr, mono=True)
    signal = signal[:]
    if timestamps:
        res_marks = {}
        for current in timestamps:
            for_volume = signal[0].tolist()
            for_volume = for_volume[current[0][1]*sr:current[1][1]*sr]
            envelope = amplitude_envelope(for_volume)
            incrs, decrs = calculate_angles(envelope)
            marks_incr, marks_decr = get_rapidness(incrs, envelope), get_rapidness(decrs, envelope)
            transormed = torch.tensor(for_volume)
            transormed.unsqueeze_(0)
            marks = (marks_incr, marks_decr) # Timestamps of increasings and decreasings
            temp = get_temp(transormed.numpy(), sr)

            # Emotions
            final_emo_score = get_emotions(transormed, sr, device, model_path)

            result = {'time_incr': marks[0], 'time_decr': marks[1],
                    'count_incr': len(marks_incr), 'count_decr': len(marks_decr),
                    'temp': temp, 'emotions': final_emo_score}
            res_marks.update({name+current[0][0]:result})

            if annotation:
                for i in marks[0]:
                    print('Громкость была резко повышена на: {} секунде'.format(current[0][1]+round(i*1024/sr, 3)))
                    print('Темп речи: {}'.format(temp))
                for d in marks[1]:
                    print('Громкость была резко понижена на: {} секунде'.format(current[0][1]+round(i*1024/sr, 3)))
                    print('Темп речи: {}'.format(temp))
    else:
        res_marks = {}
        chunks = utils.chunkizer(duration, signal[0], sr)
        time = 0
        for chunk in chunks:
            envelope = amplitude_envelope(chunk)
            incrs, decrs = calculate_angles(envelope)
            marks_incr, marks_decr = get_rapidness(incrs, envelope), get_rapidness(decrs, envelope)
            transormed = torch.tensor(chunk)
            transormed.unsqueeze_(0)
            marks = (marks_incr, marks_decr) # Timestamps of increasings and decreasings
            temp = get_temp(transormed.numpy(), sr)

            # Emotions
            final_emo_score = get_emotions(transormed, sr, device)

            result = {'time_incr': marks[0], 'time_decr': marks[1],
                    'count_incr': len(marks_incr), 'count_decr': len(marks_decr),
                    'temp': temp, 'emotions': final_emo_score}
            res_marks.update({name+str(time):result})

            if annotation:
                for i in marks[0]:
                    print('Громкость была резко повышена на: {} секунде'.format(time+round(i*1024/sr, 3)))
                    print('Темп речи: {}'.format(temp))
                for d in marks[1]:
                    print('Громкость была резко понижена на: {} секунде'.format(time+round(i*1024/sr, 3)))
                    print('Темп речи: {}'.format(temp))
            time += duration
    return res_marks


if __name__=="__main__":
    result = sound_markers('example_part.mp4')
    print(result)

    # print(m, li, ld, t)
    

#!/usr/bin/env python3

from py import process
from vosk import Model, KaldiRecognizer, SetLogLevel
import sys
import os
import subprocess
import re
import torch

# Model for punctuation recognition.
punct_model, _, _, _, apply_text = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_te')

def processing(raw_text):
    r = re.compile("[а-яА-Я]+")

    raw_text = ' '.join(raw_text)
    raw_text = raw_text.replace('"', '')
    words = raw_text.split(' ')

    words = [w for w in filter(r.match, words)]

    # Punctuation
    text = ' '.join(words)
    text = text.replace('\n}', '')

    return apply_text(text, lan='ru')


def get_raw(path, model_path="/home/lameus/vosk-api/python/example/model"):

    SetLogLevel(0)


    sample_rate=16000
    model = Model(model_path)
    rec = KaldiRecognizer(model, sample_rate)

    process = subprocess.Popen(['ffmpeg', '-loglevel', 'quiet', '-i',
                                f"{path}",
                                '-ar', str(sample_rate) , '-ac', '1', '-f', 's16le', '-'],
                                stdout=subprocess.PIPE)

    result = []
    while True:
        data = process.stdout.read(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result.append(rec.Result())

    result.append(rec.FinalResult())

    return result

#a = get_raw(path='check.mp3')
#print(processing(a))

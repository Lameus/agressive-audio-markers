import math
import warnings

import librosa

warnings.filterwarnings('ignore')



# Делит аудио файл(фрагменты) на чанки
def chunkizer(chunk_length, audio, sr):
    num_chunks = math.ceil(librosa.get_duration(audio, sr=sr) / chunk_length)
    chunks = []
    for i in range(num_chunks):
        chunks.append(audio[i * chunk_length * sr:(i + 1) * chunk_length * sr])
    return chunks



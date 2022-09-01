def chunkizer(chunk_length, audio, sr):
    duration = audio.shape[0] / sr
    num_chunks = int(-(-duration // chunk_length)) # ceil without import math
    chunks = []
    for i in range(num_chunks):
        chunks.append(audio[i * chunk_length * sr:(i + 1) * chunk_length * sr])
    return chunks

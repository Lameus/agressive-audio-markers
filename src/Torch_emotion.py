import torch
import torchaudio
import numpy as np
from src.utils import chunkizer

from src.AudioModel import AudioModel


class Torch_emotion:
    # Заменить audio_path на False и по умолчанию передавать chunks
    def __init__(self, model, signal, device='cpu', sr=44100):
        self.signal = signal
        self.device = device

        self.model = model
        self.sr = sr
        self.num_samples = 3 * sr
        self.target_sample_rate = sr

        self.predict = []  # Значения эмоций

    def audio_pipeline(self):
        # self.signal = self.to_voice(self.signal)
        self.chunks = chunkizer(3, self.signal[0].numpy(), self.sr)
        self.chunks = [torch.Tensor(i) for i in self.chunks]  # Перевод фрагментов в тензоры
        self.test = []  # Лист для записи преобразованных фрагментов
        for i in range(len(self.chunks)):
            # Преобразование к одному значению sample rate
            w = self.chunks[i]
            w.unsqueeze_(0)
            w = self._cut_if_necessary(w)
            w = self._right_pad_if_necessary(w)

            mfcc = torchaudio.transforms.MFCC(sample_rate=self.sr,
                                              n_mfcc=13)(w)

            mfcc = np.transpose(mfcc.numpy(), (1, 2, 0))
            mfcc = np.transpose(mfcc, (2, 0, 1)).astype(np.float32)

            self.test.append(torch.tensor(mfcc, dtype=torch.float))

        # Предсказание
        self.model.eval()
        for i in range(len(self.test)):
            c = self.test[i]
            c.unsqueeze_(0)
            # Добавление значения в список предсказаний
            self.predict.append(self.model(c).to('cpu').detach().numpy()[0])
            

        return self.predict

    # Функции обработки аудио
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal


# if __name__ == '__main__':
#     Test = Torch_emotion(audio_path='ю/0.0_0.0.mp3')
#     print(Test.audio_pipeline())

from functools import partial

import librosa
import numpy as np
import scipy


class Sequential:
    def __init__(self, *args):
        self.transforms = args

    def __call__(self, inp: np.ndarray):
        res = inp
        for transform in self.transforms:
            res = transform(res)
        return res


class Windowing:
    def __init__(self, window_size=1024, hop_length=None):
        self.window_size = window_size
        self.hop_length = hop_length if hop_length else self.window_size // 2
    
    def __call__(self, waveform):
        # Your code here
        size = waveform.shape[0]
        # print(waveform.shape, (waveform.shape[0] - self.window_size % 2) // self.hop_length + 1, end=' ')
        
        waveform = np.concatenate([
            np.zeros(self.window_size // 2), 
            waveform, 
            np.zeros(self.window_size // 2)
        ])
        # print(self.window_size // 2, waveform)
        
        windows = []
        for i in range(0, size - self.window_size % 2 + 1, self.hop_length):
            windows.append(waveform[i: i + self.window_size])
            # print(windows[-1])
        
        return np.stack(windows, axis=0)
    

class Hann:
    def __init__(self, window_size=1024):
        # Your code here
        self.window_size = window_size

    
    def __call__(self, windows):
        # Your code here
        for i in range(windows.shape[0]):
            windows[i] = windows[i] * scipy.signal.windows.hann(self.window_size, sym=False)

        return windows

class DFT:
    def __init__(self, n_freqs=None):
        self.n_freqs = n_freqs

    def __call__(self, windows):
        # Your code here
        spec = []
        for window in windows:
            spec.append(np.fft.rfft(window))

        out = np.absolute(np.stack(spec, axis=0)[:, :self.n_freqs])

        return out


class Square:
    def __call__(self, array):
        return np.square(array)


class Mel:
    def __init__(self, n_fft, n_mels=80, sample_rate=22050):
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate

    def __call__(self, spec):
        filter = librosa.filters.mel(
            sr=self.sample_rate, 
            n_fft=self.n_fft, 
            n_mels=self.n_mels,
            fmin=1, 
            fmax=8192
        )

        return np.dot(spec, filter.T)

    def restore(self, mel):
        filter = librosa.filters.mel(
            sr=self.sample_rate, 
            n_fft=self.n_fft, 
            n_mels=self.n_mels, 
            fmin=1, 
            fmax=8192
        )
        print(mel.shape, np.linalg.pinv(filter).shape)
        spec = np.dot(mel, np.linalg.pinv(filter).T)

        return spec


class GriffinLim:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.griffin_lim = partial(
            librosa.griffinlim,
            n_iter=32,
            hop_length=hop_length,
            win_length=window_size,
            n_fft=window_size,
            window='hann'
        )

    def __call__(self, spec):
        return self.griffin_lim(spec.T)


class Wav2Spectrogram:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.windowing = Windowing(window_size=window_size, hop_length=hop_length)
        self.hann = Hann(window_size=window_size)
        self.fft = DFT(n_freqs=n_freqs)
        # self.square = Square()
        self.griffin_lim = GriffinLim(window_size=window_size, hop_length=hop_length, n_freqs=n_freqs)

    def __call__(self, waveform):
        return self.fft(self.hann(self.windowing(waveform)))

    def restore(self, spec):
        return self.griffin_lim(spec)


class Wav2Mel:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None, n_mels=80, sample_rate=22050):
        self.wav_to_spec = Wav2Spectrogram(
            window_size=window_size,
            hop_length=hop_length,
            n_freqs=n_freqs)
        self.spec_to_mel = Mel(
            n_fft=window_size,
            n_mels=n_mels,
            sample_rate=sample_rate)

    def __call__(self, waveform):
        return self.spec_to_mel(self.wav_to_spec(waveform))

    def restore(self, mel):
        return self.wav_to_spec.restore(self.spec_to_mel.restore(mel))


class TimeReverse:
    def __call__(self, mel):
        return mel[::-1]

class Loudness:
    def __init__(self, loudness_factor):
        self.loundness_factor = loudness_factor

    def __call__(self, mel):
        return mel * self.loundness_factor

class PitchUp:
    def __init__(self, num_mels_up):
        self.num_mels_up = num_mels_up

    def __call__(self, mel):
        pass
        

class PitchDown:
    def __init__(self, num_mels_down):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^



class SpeedUpDown:
    def __init__(self, speed_up_factor=1.0):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^



class FrequenciesSwap:
    def __call__(self, mel):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^



class WeakFrequenciesRemoval:
    def __init__(self, quantile=0.05):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^



class Cringe1:
    def __init__(self):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^



class Cringe2:
    def __init__(self):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


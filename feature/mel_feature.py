import numpy as np
from scipy.fftpack import fft
from scipy import signal
from librosa.feature import melspectrogram

class MelFeature():
    '''
    该方法实现的Mel特征总觉得不对，停止使用
    改为使用MelFeature2/MelFeature3
    '''
    def __init__(self,sr = 16000,n_fft=2048,hop_length=512,power=2.0,n_mels = 200,**kwargs):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.n_mels = n_mels

    def batch_mfcc(self, x):
        '''
        先提取特征
        :param x:
        :param kwargs:
        :return:
        '''

        features = list(map(lambda x: self.mfcc(x), x))

        return features

    def __call__(self,x):
        return self.batch_mfcc(x)

    def mfcc(self,x):
        '''
        没有padding
        :param x:
        :return:
        '''
        x = x / np.max(np.abs(x)) # 音量归一化

        return melspectrogram(x,
                              sr=self.sr,
                              n_fft=self.n_fft,
                              hop_length=self.hop_length,
                              power=self.power,
                              n_mels = self.n_mels)

class MelFeature2():
    '''
    参考该博客的实现，较为简单，但是不容易控制特征长度
    https://www.kaggle.com/ybonde/log-spectrogram-and-mfcc-filter-bank-example
    '''
    def __init__(self,sr = 16000,window_size = 25,step_size = 8):
        self.sr = sr
        self.nperseg = int(round(window_size * sr / 1e3))
        self.noverlap = int(round(step_size * sr / 1e3))

    def batch_mfcc(self,x):
        '''
        :param x: (batch,timestamp)
        :return: (batch,feature_dim,timestamp)
        '''
        features = list(map(lambda x: self.mfcc(x), x))
        return features

    def __call__(self, x):
        return self.batch_mfcc(x)

    def mfcc(self,x):
        '''
        :param x: (timestamp,)
        :return: (feature_dim,timestamp)
        '''
        x = x/np.max(np.abs(x))

        _, _, spec = signal.spectrogram(x, fs=self.sr,
                                        window='hann',
                                        nperseg=self.nperseg, noverlap=self.noverlap,
                                        detrend=False)
        eps = 1e-10
        return np.log(spec.T.astype(np.float32) + eps).T

class MelFeature3():
    '''
    参考该博客的实现，博客介绍的较为具体，可以参考
    https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    '''
    def __init__(self, sr=16000,pre_emphasis = 0.97,n_mels = 200):
        self.sr = sr

        self.pre_emphasis = pre_emphasis

        self.frame_size = 0.05
        self.frame_stride = 0.01

        self.frame_length = self.frame_size * self.sr   # Convert from seconds to samples
        self.frame_step = self.frame_stride * self.sr
        self.frame_length = int(round(self.frame_length))
        self.frame_step = int(round(self.frame_step))



        self.NFFT = 512 # typically 256 or 512
        self.nfilt = n_mels

        self.fbank = np.zeros((self.nfilt, int(np.floor(self.NFFT / 2 + 1))))
        self._initial_filter_bank()

    def batch_mfcc(self, x):
        features = list(map(lambda x: self.mfcc(x), x))
        return features

    def _initial_filter_bank(self):
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (self.sr / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = np.floor((self.NFFT + 1) * hz_points / self.sr)

        for m in range(1, self.nfilt + 1):
            f_m_minus = int(bin[m - 1])  # left
            f_m = int(bin[m])  # center
            f_m_plus = int(bin[m + 1])  # right

            for k in range(f_m_minus, f_m):
                self.fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                self.fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    def __call__(self, x):
        return self.batch_mfcc(x)

    def _pre_emphasis(self, x):
        emphasized_signal = np.append(x[0], x[1:] - self.pre_emphasis * x[:-1])
        return emphasized_signal

    def _framing(self,x):
        '''

        :param x: audio signal after self._pre_emphasis(x)
        :return:
        '''

        signal_length = len(x)

        num_frames = int(np.ceil(float(np.abs(signal_length - self.frame_length)) / self.frame_step))  # Make sure that we have at least 1 frame

        pad_signal_length = num_frames * self.frame_step + self.frame_length
        z = np.zeros((pad_signal_length - signal_length))

        # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
        pad_signal = np.append(x,z)

        indices = np.tile(np.arange(0, self.frame_length), (num_frames, 1)) + np.tile(
            np.arange(0, num_frames * self.frame_step, self.frame_step), (self.frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]

        return frames

    def _ftt(self,x):
        '''
        :param x: audio signal after self._framing(x)
        :return:
        '''
        x *= np.hamming(self.frame_length)
        mag_frames = np.absolute(np.fft.rfft(x, self.NFFT))  # Magnitude of the FFT
        pow_frames = ((1.0 / self.NFFT) * ((mag_frames) ** 2))  # Power Spectrum
        return pow_frames

    def _filter_bank(self,x):
        '''

        :param x: after self._ftt(x)
        :return:
        '''
        filter_banks = np.dot(x, self.fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)  # dB

        return filter_banks.T

    def mfcc(self, x):
        '''
        :param x: [batch,feature_dim,time_stamp]
        :return:
        '''
        ex_flow = [self._pre_emphasis,self._framing,self._ftt,self._filter_bank ]

        for ifun in ex_flow:
            x = ifun(x)

        return x

class MelFeature4():
    def __init__(self,fs = 16000,window_length = 400,n_mels = 200,coverge = 0.5,fixed = False):
        self.x = np.linspace(0, 400 - 1, 400, dtype=np.int64)

        self.time_window = 25  # 单位ms
        self.window_length = window_length  # 计算窗长度的公式，目前全部为400固定值
        self.w = np.hamming(self.window_length)
        self.fs = fs
        self.n_mels = n_mels
        self.coverge = coverge
        self.fixed = fixed

    def batch_mfcc(self,x:np.ndarray):
        features = list(map(lambda x: self.mfcc(x), x))
        return features

    def mfcc(self,x:np.ndarray):
        '''
        :param x: (len,)
        :return:
        '''

        wav_length = x.shape[0]


        pad_wav_length = int(wav_length + (self.window_length*self.coverge) - (wav_length - self.window_length) % (self.window_length*self.coverge))

        x = np.pad(x,(0,pad_wav_length-wav_length),mode="constant",constant_values = 0)

        range_count = int((pad_wav_length-self.window_length)/(self.window_length*self.coverge)+1)
        if self.fixed:
            int(wav_length/ self.fs* 1000 - 25) // 10  # 计算循环终止的位置，也就是最终生成的窗数

        data_input = np.zeros((range_count, self.n_mels), dtype=np.float)  # 用于存放最终的频率特征数据
        # data_line = np.zeros((1, 400), dtype=np.float)
        for i in range(0, range_count-1):
            p_start = i * int(self.window_length * self.coverge)

            if self.fixed:
                p_start = i * 160
            p_end = p_start + self.window_length

            data_line = x[p_start:p_end]

            data_line = data_line * self.w  # 加窗

            data_line = np.abs(fft(data_line)) / wav_length

            data_input[i] = data_line[0:self.n_mels]  # 设置为400除以2的值（即200）是取一半数据，因为是对称的

        data_input = np.log(data_input + 1)

        return data_input.T

class MelFeature5():
    '''目前效果最好的特征提取方法，暂时不清楚原因'''
    def __init__(self):
        self.w = np.hamming(400)
        self.fs = 16000
        self.time_window = 25

    def batch_mfcc(self,x:np.ndarray):
        features = list(map(lambda x: self.mfcc(x), x))
        return features

    def mfcc(self, x):
        '''
        :param x: [1,vector]
        :param fs:
        :return:
        '''
        wav_length = x.shape[0]

        range0_end = int(wav_length / self.fs * 1000 - self.time_window) // 10  # 计算循环终止的位置，也就是最终生成的窗数
        data_input = np.zeros((range0_end, 200), dtype=np.float)  # 用于存放最终的频率特征数据

        for i in range(0, range0_end):
            p_start = i * 160
            p_end = p_start + 400

            data_line = x[p_start:p_end]

            data_line = data_line * self.w  # 加窗

            data_line = np.abs(fft(data_line)) / wav_length

            data_input[i] = data_line[0:200]  # 设置为400除以2的值（即200）是取一半数据，因为是对称的

        data_input = np.log(data_input + 1)
        return data_input.T
import os,math
import librosa

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile


class Recoder():
    @staticmethod
    def record(second = 5,chunk = 256,sample_rate = 16000,channels = 1,format = None):
        import pyaudio

        frames = []
        if format is None:
            format = pyaudio.paInt16

        p = pyaudio.PyAudio()
        stream = p.open(format=format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk)

        print(f"[info*]开始录音：请在 {second} 秒内输入语音")
        frame_range = int(sample_rate / chunk * second)
        for i in range(0, frame_range):
            data = stream.read(chunk)
            data = np.frombuffer(data,dtype=np.short)
            frames.extend(data)
        print("[info*]录音结束\n")

        stream.stop_stream()
        stream.close()
        p.terminate()

        return np.array(frames,dtype=np.short)

class NoiseFilter():
    def _nextpow2(self, a):
        '''找到最接近a的2的n次幂的n'''
        a = abs(a)
        time = 0
        rval = 1
        while rval < a:
            rval <<= 1
            time += 1
        return time

    def _berouti(self, SNR, a=4):
        if -5.0 <= SNR <= 20.0:
            ele = a - SNR * (a-1) / 20
        elif SNR < -5.0:
            ele = a+1
        else:
            ele = 1
        return ele

    def noise_filter(self,x,fs=16000):
        # 计算参数
        window_length = 20 * fs // 1000 # 样本中帧的大小
        PERC = 50 # 窗口重叠占帧的百分比
        cover_window = window_length * PERC // 100  # 重叠窗口
        uncover_window = window_length - cover_window   # 非重叠窗口
        # 设置默认参数
        Thres = 3
        Expnt = 2.0
        beta = 0.002
        G = 0.9
        # 初始化汉明窗
        win = np.hamming(window_length)
        # normalization gain for overlap+add with 50% overlap
        winGain = uncover_window / sum(win)

        # Noise magnitude calculations - assuming that the first 5 frames is noise/silence
        nFFT = 2 * 2 ** (self._nextpow2(window_length))
        noise_mean = np.zeros(nFFT)

        j = 0
        for k in range(1, 6):
            noise_mean = noise_mean + abs(np.fft.fft(win * x[j:j + window_length], nFFT))
            j = j + window_length
        noise_mu = noise_mean / 5

        # --- allocate memory and initialize various variables
        k = 1
        img = 1j
        x_old = np.zeros(cover_window)
        Nframes = len(x) // uncover_window - 1
        xfinal = np.zeros(Nframes * uncover_window)

        # =========================    Start Processing   ===============================
        for n in range(0, Nframes):
            # Windowing
            insign = win * x[k-1:k + window_length - 1]
            # compute fourier transform of a frame
            spec = np.fft.fft(insign, nFFT)
            # compute the magnitude
            sig = abs(spec)

            # save the noisy phase information
            theta = np.angle(spec)
            SNRseg = 10 * np.log10(np.linalg.norm(sig, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)


            if Expnt == 1.0:  # 幅度谱
                alpha = self._berouti(SNRseg, 3)
            else:  # 功率谱
                alpha = self._berouti(SNRseg, 4)

            sub_speech = sig ** Expnt - alpha * noise_mu ** Expnt
            # 当纯净信号小于噪声信号的功率时
            diffw = sub_speech - beta * noise_mu ** Expnt
            # beta negative components

            z = [i for i,frame in enumerate(diffw) if frame<0]

            if len(z) > 0:
                sub_speech[z] = beta * noise_mu[z] ** Expnt
            if SNRseg < Thres:  # Update noise spectrum
                noise_temp = G * noise_mu ** Expnt + (1 - G) * sig ** Expnt  # 平滑处理噪声功率谱
                noise_mu = noise_temp ** (1 / Expnt)  # 新的噪声幅度谱
            # flipud函数实现矩阵的上下翻转，是以矩阵的“水平中线”为对称轴
            # 交换上下对称元素
            sub_speech[nFFT // 2 + 1:nFFT] = np.flipud(sub_speech[1:nFFT // 2])
            x_phase = (sub_speech ** (1 / Expnt)) * (np.array([math.cos(x) for x in theta]) + img * (np.array([math.sin(x) for x in theta])))
            # take the IFFT

            xi = np.fft.ifft(x_phase).real
            # --- Overlap and add ---------------
            xfinal[k-1:k + uncover_window - 1] = x_old + xi[0:cover_window]
            x_old = xi[0 + cover_window:window_length]
            k = k + uncover_window

        return np.array(xfinal*winGain,dtype=np.short)

class VadExtract():
    frame_ranges = []
    sample_ranges = []

    def __init__(self,sr = 16000,hop_length = 256,tail_length = 20,merge_dist = 20):
        self.sr = sr
        self.hop_length = hop_length
        self.tail_length = tail_length # 截取后在前后多截取一部分
        self.merge_dist = merge_dist # 帧左右合并的扩张阈值

    def merge(self,arange):
        '''
        合并所有的range，返回一个[start,end]
        :param arange: [(s,e),(s,e),(s,e)...]，必须保证是按序给出
        :return:
        '''
        return [arange[0][0],arange[-1][-1]]

    def extract_audio(self, y):
        fr,vr,_ = self.extract(y)
        vr = self.merge(vr)
        return y[vr[0]:vr[1]]

    def audio2batch_by_extract(self,y):
        _,vr,_ = self.extract(y)
        return self._extract_part_audio(y, vr)

    def _extract_part_audio(self, y, vr):
        ys = [y[svr[0]:svr[1]] for svr in vr]
        return np.array(ys)

    def extract_part_audio_and_merge(self,y):
        '''
        每一段的都抽取出来，然后再合并，将对话中间的空缺（如有）去掉
        :param y:
        :return:
        '''
        fr,vr,_ = self.extract(y)
        ys = self._extract_part_audio(y, vr)
        y = np.concatenate(ys)
        return y

    def batch_extract(self,batch,merge = True):
        '''

        :param batch: 多个y [sample, num_y]
        :param merge: 是否拼合
        :return:
            返回一个 多维的list
            如果 merge = True，那么 [sample,2]
                 merge = False,那么 [sample,n,2],n是每段音频切分出来的结果
        '''
        batch = [self.extract(y)[1] for y in batch] # 注意返回的是三个参数，这里只取下标
        if merge:
            batch = [self.merge(y) for y in batch]

        return batch

    def cut_audio(self,y,vector_range):
        pass # TODO

    def batch_cut_audio(self,batch,batch_range):
        '''
        :param batch:
        :param batch_range:
        :return:
        '''

    def extract(self, y):
        '''
        抽取主方法
        :param y: 采样序列
        :return:
            frame_ranges ： 包含[(start,end),...]的帧的list
            vector_ranges ： 包含 [(start,end),...]的向量下标的list
            param_dict ： 包含抽取过程中的参数，调用plot函数时传入
        '''
        y = y.astype(np.float32)
        rms = librosa.feature.rmse(y=y, hop_length=self.hop_length)  # rms envelope
        max_rms = np.max(rms)  # 求个最大值做特征
        length = rms.shape[0]  # 总个数
        avg_rms = np.mean(rms)
        rms_thresh = max_rms / 10  # 暂定阈值

        param_dict = dict(
            rms = rms,
            rms_thresh = rms_thresh,
            max_rms = max_rms,
            length = length,
            avg_rms = avg_rms,
        )

        active_frames = [frame for frame,element in enumerate(rms.T) if element>rms_thresh] #根据阈值过滤帧
        frame_ranges = self.frame2range(active_frames)  # 合并帧，得到相应的范围

        vector_ranges = [librosa.frames_to_samples(speech_range, hop_length=self.hop_length).tolist() # 帧转化为采样，hop_length要与截取帧时相同
                            for speech_range in frame_ranges]

        return frame_ranges,vector_ranges,param_dict


    def write(self, fdir, fbase_name, y, sample_ranges):
        '''
        输出相应的截取音频
        :param fdir:
        :param fbase_name:
        :param y:
        :param sample_ranges:
        :return:
        '''
        os.makedirs(fdir,exist_ok=True)
        fpre, ext = os.path.splitext(fbase_name)
        for i, sample_range in enumerate(sample_ranges):
            sub_y = y[sample_range[0]:sample_range[1]+self.tail_length]
            wavfile.write(os.path.join(f"{fpre}-{i}.{ext}"), self.sr, sub_y)


    def frame2range(self, frames):
        '''
        将整串帧简化为帧的区间，返回区间的集合
        :param frames: 筛选后的所有帧序列
        :return: [[28, 329], [351, 439]] ，表示帧，要截取还需要通过采样率转化为相应的下标
        '''
        range_result = []
        size = len(frames)

        i = 0
        while i < size :
            start_frame = frames[i]
            while i<size-1  and frames[i + 1]-frames[i] < self.merge_dist:
                i += 1
            end_frame = frames[i]

            range_result.append([start_frame, end_frame])
            i += 1

        return range_result

    def plot(self,frame_range,param_dict):
        plt.figure()
        plt.subplot(1, 1, 1)
        # 辅助线

        plt.hlines(param_dict["rms_thresh"], 0, param_dict["length"],
                   colors='g', label='THRESH')  # 阈值线
        plt.hlines(param_dict["avg_rms"], 0, param_dict["length"],colors='b', label='AVG')  # 平均值线
        for speech_range in frame_range:
            plt.vlines(
                range(speech_range[0], speech_range[1], 10), 0, param_dict["max_rms"],
                colors="y")  # 内部每10帧一条黄色辅助线
            plt.vlines(speech_range[0], 0, param_dict["max_rms"], colors="r")  # 区间分界线
            plt.vlines(speech_range[1] + self.tail_length, 0, param_dict["max_rms"], colors="r")

        # 回归至采样编号，目的是能够写入wav，测试效果
        # 测试文件："z200\G0002\session01\T0055G0002S0424.wav"
        # 起始帧编号
        # 预计值：0.5*16000=8000
        # 实际值：7168
        # 结束帧编号
        # 预计值：7*16000=112000
        # 实际值：112384

        plt.semilogy(param_dict["rms"].T, label='RMS Energy')
        plt.xticks([])
        plt.xlim([0, param_dict["rms"].shape[-1]])
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
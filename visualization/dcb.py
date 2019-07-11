import pyaudio
import wave
import time
import os
dir_path = os.path.split(os.path.realpath(__file__))[0] #"./util"

class proc:
    sign = 0  # 暂停标志
    fini = 0  # 完成标志
    rereco = 0  # 重读标志
    ending = 0  # 结束标志
    filepath = os.path.join(dir_path,"cache.wav")   # cheche3是暂停之间的音频缓存
    flu = 0  # 识别标志

    def get_audio(self, ab):
        self.ending = 0
        self.fini = 0
        self.sign = 0
        CHUNK = 256
        FORMAT = pyaudio.paInt16
        CHANNELS = 1  # 声道数
        RATE = 16000  # 采样率
        RECORD_SECONDS = 5
        WAVE_OUTPUT_FILENAME = self.filepath
        p = pyaudio.PyAudio()
        self.frames = []
        self.frames2 = []
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        tick1 = time.time()
        print("开始录音")
        # tick2 = time.time()
        while self.fini == 0:
            data = stream.read(CHUNK)
            # print(f"\r{data}",sep="\0",flush=True)
            if self.sign == 0:
                self.ending = 1
                # tick3 = int(tick2 - tick1)
                # for i in range(0, int(RATE / CHUNK * tick3)):
                self.frames.append(data)
                self.frames2.append(data)
                # stream.stop_stream()
                # stream.close()
                # wf = wave.open("./cache3.wav", 'wb')
                # wf.setnchannels(CHANNELS)
                # wf.setsampwidth(p.get_sample_size(FORMAT))
                # wf.setframerate(RATE)
                # wf.writeframes(b''.join(self.frames2))
                # wf.close()
                # while self.sign != 0:
                #     pass
                self.ending = 0
                # stream = p.open(format=FORMAT,
                #                 channels=CHANNELS,
                #                 rate=RATE,
                #                 input=True,
                #                 frames_per_buffer=CHUNK)
                # data = stream.read(CHUNK)
                # if self.flu == 1:
                #     self.frames2 = []
                # self.flu == 0

        # tick1 = time.time()
        tick2 = time.time()
        tick3 = int(tick2 - tick1)
        print(tick3)
        # for i in range(0, int(RATE / CHUNK * tick3)):
        #     self.frames.append(data)
        #     self.frames2.append(data)

        print("录音结束\n")
        self.ending = 1
        stream.stop_stream()
        stream.close()
        p.terminate()



        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames2))
        wf.close()

    def cut_audio(self):

        self.sign = 1
        print(self.sign)

    def cont_audio(self):
        self.sign = 0
        print(self.sign)

    def fini_audio(self):
        self.fini = 1
        self.sign = 0

    def recc(self):
        self.flu = 1
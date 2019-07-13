'''
用于读取音频和标签文件
'''
from matplotlib import pyplot as plt
from scipy.io import wavfile
import numpy as np
import re
import random,time,os
from math import floor

from feature.mel_feature import MelFeature2
from util.audiotool import VadExtract

from keras.utils import Sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

filter_char = list('''[!"'\(\),\.\?@q~“”… 　、。」！（），？Ａａｂｃｋｔ]*''')
# re_unchs = re.compile() # 非汉字，无拼音的




class VoiceDataGenerator():
    def __init__(self,path):
        self.set_path(path)

    def set_path(self,path):
        self.path = path

    def load_from_path(self,choose_x = True,choose_y = True):
        '''
        :param choose_x: 是否选择相应的音频文件
        :param choose_y: 是否选择相应的标注文件
        :return: x_sets,y_sets，注意当相应的choose为False时，返回的是None，但是仍然是返回两个元素，注意接收变量
        '''
        pass

    def _choose(self,x_set,y_set,choose_x,choose_y):
        if choose_x and not choose_y:
            return x_set,None
        if choose_y and not choose_x:
            return None,y_set
        return x_set,y_set

class Thchs30(VoiceDataGenerator):
    def load_from_path(self,choose_x = True,choose_y = True):
        path = os.path.abspath(self.path)
        datapath = os.path.join(path,"data") # ./data/

        assert os.path.exists(datapath),"path not exists!"

        fs = os.listdir(datapath) # .wav / .trn
        fs = [os.path.join(datapath,i) for i in fs]

        x_set = [i for i in fs if i.endswith(".wav")]
        y_set = [f"{i}.trn" for i in x_set]

        return self._choose(x_set,y_set,choose_x,choose_y)

class Z200(VoiceDataGenerator):
    def load_from_path(self,choose_x = True,choose_y = True):
        path = os.path.abspath(self.path)
        root = os.listdir(path)
        root = [os.path.join(path,i) for i in root]
        root = [os.path.join(i,"session01") for i in root if os.path.isdir(i)]
        fs = []

        for sub_dir in root:
            sub_fs = os.listdir(sub_dir)
            sub_fs = [os.path.join(sub_dir,i[:-4]) for i in sub_fs if i.endswith(".wav")]
            fs.extend(sub_fs)

        x_set = [f + ".wav" for f in fs]
        y_set = [f + ".txt" for f in fs]

        return self._choose(x_set, y_set, choose_x, choose_y)

class Primewords(VoiceDataGenerator):
    def load_from_path(self,choose_x = True,choose_y = True):
        audio_root_dir = os.path.join(self.path,"audio_files") # ./audio_files/
        f0 = os.listdir(audio_root_dir) # 从0-f 的16个目录
        f0 = [os.path.join(audio_root_dir,f) for f in f0]

        ff00 = []  # 包含所有00-ff的文件夹路径
        for f in f0:
            subff = os.listdir(f) # 列出00-ff几个文件夹
            subff = [os.path.join(f,sf) for sf in subff]
            ff00.extend(subff)

        allfs = []
        for subff in ff00:
            wavfs = os.listdir(subff)
            wavfs = [os.path.join(subff,f[:-4]) for f in wavfs if f.endswith(".wav")]
            allfs.extend(wavfs)
        x_set = [f"{f}.wav" for f in allfs]
        y_set = [f"{f}.txt" for f in allfs]

        return self._choose(x_set, y_set, choose_x, choose_y)

class ST_CMDS(VoiceDataGenerator):
    def load_from_path(self,choose_x = True,choose_y = True):
        fs = os.listdir(self.path)
        allfs = [os.path.join(self.path, f[:-4]) for f in fs if f.endswith(".txt")]

        x_set = [f"{f}.wav" for f in allfs]
        y_set = [f"{f}.txt" for f in allfs]

        return self._choose(x_set, y_set, choose_x, choose_y)

class AiShell(VoiceDataGenerator):
    def load_from_path(self,choose_x = True,choose_y = True):
        train_root = os.path.join(self.path, "wav/train")
        test_root = os.path.join(self.path, "wav/test")
        dev_root = os.path.join(self.path, "wav/dev")

        allfs = []
        for fs in [train_root, test_root, dev_root]:  # 每一个目录,可以去掉test和dev，只保留训练集，但没啥必要
            fs = self._get_sub_wavs(fs)
            allfs.extend(fs)

        x_set = [f"{f}.wav" for f in allfs]
        y_set = [f"{f}.txt" for f in allfs]

        return self._choose(x_set, y_set, choose_x, choose_y)

    def _get_sub_wavs(self,path):
        '''
        接收 ./train/  ./dev/ ./test ,返回相应路径下所有的wav文件
        :param path:
        :return:
        '''
        s_fs = os.listdir(path)
        fs = []
        for f in s_fs:
            s_path = os.path.join(path,f)
            wavfs = os.listdir(s_path)
            wavfs = [os.path.join(s_path,wavf) for wavf in wavfs]
            fs.extend(wavfs)

        fs = [f[:-4] for f in fs if f.endswith(".wav")] # 过滤审查

        return fs

class Currentpath(VoiceDataGenerator):
    '''
    从当前目录下选取所有的wav文件和txt文件
    '''
    def load_from_path(self,choose_x = True,choose_y = True):
        fs = os.listdir(self.path)
        fs = [os.path.join(self.path,f) for f in fs]
        fs = [f[:-4] for f in fs if f.endswith(".wav")]

        x_set = [f"{f}.wav" for f in fs]
        y_set = [f"{f}.txt" for f in fs]
        return self._choose(x_set, y_set, choose_x, choose_y)

class TextDataGenerator():
    def __init__(self,root_path):
        self.path = root_path

    def load_from_path(self):
        dir_iter = os.walk(self.path)
        txtfs = []
        for path, d, filelist in dir_iter:
            for filename in filelist:
                txtfs.append(os.path.join(path, filename))

        return txtfs

class VoiceDatasetList():
    def merge_load(self,lst:list,choose_x = True,choose_y = True):
        '''
        合并多个数据集
        :param lst: DataGenerator类
        :param choose_x: 是否读取音频，用于声学模型
        :param choose_y: 是否读取文本，用于声学模型和语言模型
        :return: x_sets,y_sets，注意当相应choose为False时，返回的list无元素
        '''
        x_sets,y_sets = [],[]
        for gene in lst:
            x_set,y_set = gene.load_from_path(choose_x,choose_y)
            if choose_x:
                x_sets.extend(x_set)
            if choose_y:
                y_sets.extend(y_set)
        return x_sets,y_sets

class DataLoader(Sequence):
    '''TODO 找机会将VoiceLoader 和 TextLoader重构一下，两者有比较多的可以重合的地方'''

    train_mode = "train"
    test_mode = "test"
    evlu_mode = "evaluate"
    def __init__(self,batch_size = 32) -> None:
        self.batch_size = batch_size

    def get_item(self, i, batch_size=None, mode="train"):
        pass

    def __getitem__(self, item):
        return self.get_item(item, batch_size=self.batch_size, mode="train")

class VoiceLoader(DataLoader):
    '''
    根据路径，加载音乐，同时提取特征,继承自Sequence，从而保证了线程安全
    标注必须保证分2-3行，第一行必须是汉字，第二行必须是拼音，第三行是音素（如有）
    '''
    def __init__(self, x_set, y_set=None,
                 pymap=None,
                 batch_size = 32, n_mels = 128, feature_pad_len=256, max_label_len=32,feature_dim = 2,
                 cut_sub = None,vad_cut = True,
                 melf = None,
                 check = True,
                 padding = True,
                 divide_feature_len = 1,
                 create_for_train = True,
                 all_train = True,
                 test_set_rate = 0.1,
                 evlu_set_rate = 0.1):
        '''
        用于训练和测试和预测皆可，实现了所有相应的方法
        :param x_set:音乐的文件路径list ，要求采样率均为 16000hz
        :param y_set: 标签的文件路径list，保证是npy格式，并且同意按同一个拼音字典处理过
        :param hop_length: 因为更换了特征抽取的类，目前没有用
        :param n_mels: 用来验证特征向量维度，向量维度由melf的提取方法确定，在该类初始化完成后会用该数值进行验证
        :param feature_pad_len: 最大特征时间步长度，用于padding
        :param max_label_len: 最大标签长度，用于padding
        :param cut_sub: 选取子数据集进行训练，验证小数据集上的拟合效果
            注意，是先选取子数据集，再shuffle，文件的读取顺序相同则每次都会取到相同的文件
            如果有其他用途，请自行在构建类前打乱
        :param padding:是否对齐
        :param sil_mode: 用来决定blank的padding时的index，具体参考 util.pinyin_map.PinyinMapper,默认为0
        :param divide_feature_len: 如果有池化层，一层池化层相当于在原采样率的基础上再进行一次采样，因此在这里手动除长度
                该方法一来在直觉上符合特征提取规律，另一方面用于保证最大标签长度不会超过ctc解码时的特征长度
        :param create_for_train: 构建该类是否用于训练，如果为是，则要求y_set，否则y_set可以为None
        :param all_train: 在训练时，是否全部数据集都用于训练,默认为True
        :param test_set_rate: 测试集的比例，只有在all_train = False时才考虑
        :param evlu_set_rate: 验证集的比例，只有在all_train = False时才考虑
        '''
        super().__init__()
        assert len(x_set) == len(y_set), "the size of x_set not equal to y_set."
        self.x_set = x_set
        self.y_set = y_set

        if cut_sub is not None:
            self.x_set = x_set[:cut_sub]
            if create_for_train:
                self.y_set = y_set[:cut_sub]
        self.set_size = len(self.x_set)
        self.on_epoch_end()

        self.all_train = all_train
        if create_for_train and not all_train:
            self.x_set = x_set[:floor(self.set_size*(1-test_set_rate-evlu_set_rate))]
            self.y_set = y_set[:floor(self.set_size*(1-test_set_rate-evlu_set_rate))]
            self.test_x_set = x_set[floor(-self.set_size * test_set_rate):]
            self.test_y_set = y_set[floor(-self.set_size * test_set_rate):]
            self.eval_x_set = x_set[floor(-self.set_size * (test_set_rate + evlu_set_rate)):floor(-self.set_size * (test_set_rate))]
            self.eval_y_set = y_set[floor(-self.set_size * (test_set_rate + evlu_set_rate)):floor(-self.set_size * (test_set_rate))]

            self.set_size = len(self.x_set)
            self.test_set_size = len(self.test_x_set)
            self.eval_set_size = len(self.eval_x_set)

        self.on_epoch_end()

        self.create_for_train = create_for_train
        self.batch_size = batch_size
        self.feature_pad_len = feature_pad_len if padding else None
        self.max_label_len = max_label_len
        self.n_mels = n_mels

        if melf is None:
            melf = MelFeature2(window_size=35)
        self.melf = melf

        self.feature_dim = feature_dim
        self.max_audio_len = 125000

        self.vad_cut = vad_cut
        self.divide_feature_len = divide_feature_len

        if check:
            self._check_pading_avai()
        self.vad = VadExtract()

        self.pymap = pymap

        if self.pymap.sil_mode == 0:
            self.py_pad_index = 0
        elif self.pymap.sil_mode == -1:
            self.py_pad_index = 0 #self.pymap.max_index
        else:
            self.py_pad_index = 0

    def _check_pading_avai(self):
        audio = np.random.rand(self.max_audio_len)
        feature = self.melf.mfcc(audio)
        print(feature.shape)
        feature_len = feature.shape[1]//self.divide_feature_len
        assert feature_len <= self.feature_pad_len, f"feature padding len is {self.feature_pad_len},but feature timestamp is {feature_len}"
        assert feature.shape[0] == self.n_mels, f"feature dim should be {self.n_mels},but {feature.shape[0]}"

    @staticmethod
    def audio2feature(xs,feature_pad_len,divide,melf):
        '''不光是在类中用，作为静态方法也提供给联合模型使用，TODO 这里代码写的有点乱，找时间重构一些'''
        xs = melf.batch_mfcc(xs)  # [batch,feature_num,time_stamp] ，应该在padding 后转换为[batch,time_stamp,feature_num]

        feature_len = np.array(
            [x.shape[-1] // divide for x in xs])  # 在padding前先将每个样本的time求得，用于求解之后的ctc_loss

        if feature_pad_len is not None:
            xs = [pad_sequences(f, feature_pad_len,
                                dtype="float32",
                                padding="post",
                                truncating="post") for f in xs]

        xs = np.stack(xs)  # [batch,feature_num,max_time_stamp]
        xs = np.transpose(xs, [0, 2, 1])  # [batch,max_tim_stamp,feature_num]

        feature_len = np.expand_dims(feature_len, 1)
        return xs,feature_len

    def corpus_build_iter(self):
        index = 0
        while index<self.set_size:
            # with
            pass


    def get_item(self,i,batch_size = None,mode = "train"):
        '''
        :param i: start index,
        :param batch_size: batch size
        :param mode: get dataset from "train" set,"test" set,or "evaluate" set
        :return: the data from choose mode from i to i+batch_size
        '''
        if mode == "train":
            x_set,y_set,set_size = self.x_set,self.y_set,self.set_size
        elif mode == "test":
            x_set,y_set,set_size = self.test_x_set,self.test_y_set,self.test_set_size
        elif mode == "evaluate":
            x_set,y_set,set_size = self.eval_x_set,self.eval_y_set,self.eval_set_size
        else:
            # x_set,y_set,set_size = None,None,None
            assert False,f"mode must be 'train'/'test'/'evaluate', but {mode}."

        if batch_size is None:
            batch_size = self.batch_size

        xs = []
        ys = []
        while True:
            if i == set_size - 1:
                i = 0
            '''这里可以用map替代，应该能提高读取速度'''
            try:
                audio, sample_freq = self.load_audio(x_set[i], True)
                assert sample_freq == 16000, f"this model now only support 16000hz,but {sample_freq}"

                if audio.shape[-1] > self.max_audio_len:  # 对应特征抽取后为 254 （默认hop_length下）
                    # TODO 特征提取后超出最大padding长度了，因此跳过该数据，当前架构下暂时没有好的写法，因此这样写
                    i += 1
                    continue

                xs.append(audio)
                if self.create_for_train:
                    label = self.load_label_from_txt(y_set[i])
                    ys.append(label)
                i += 1
            except:
                print(f"index {i} maybe error")
                print(f"file {self.x_set[i]} or {self.y_set[i]} may have some error,please check it and fix it.")

            if i == set_size - 1:
                i = 0

            if len(xs) == batch_size:
                break

        xs,feature_len = VoiceLoader.audio2feature(xs,self.feature_pad_len,self.divide_feature_len,self.melf)

        if self.feature_dim == 3:  # 2D卷积需要扩展维度
            xs = np.expand_dims(xs, axis=3)

        label_len = None

        if self.create_for_train:  # 训练才需要标签长度和 ys
            label_len = np.array([y.shape[0] for y in ys])
            label_len = np.expand_dims(label_len, 1)

            ys = np.array(ys)  # 第一个是否代表 空白？ TODO
            # 所有的padding必须在后面，否则ctc解码会出问题
            ys = pad_sequences(ys, self.max_label_len, padding="post", truncating="post", value=self.py_pad_index)

        placehold = np.zeros_like(ys)
        if self.create_for_train:
            return [xs, ys, feature_len, label_len], placehold
        else:
            return [xs, None, feature_len, None], None


    def create_iter(self,one_batch = False):
        index = 0
        while True:
            yield self.__getitem__(index)
            index += self.batch_size
            if index > self.set_size - 1:
                index = 0
                if one_batch:
                    break
                self.on_epoch_end()


    def choice(self):
        i = random.randint(0,self.set_size-1)
        return self.get_item(i,batch_size=self.batch_size)

    def choice_test(self):
        if self.all_train:
            return self.choice()

        i = random.randint(0,self.test_set_size-1)
        return self.get_item(i,batch_size=self.batch_size,mode="test")

    def choice_eval(self):
        if self.all_train:
            return self.choice()

        i = random.randint(0,self.eval_set_size-1)
        return self.get_item(i,batch_size=self.batch_size,mode="evaluate")

    def __len__(self):
        return self.set_size

    def on_epoch_end(self):
        c = list(zip(self.x_set, self.y_set))
        random.shuffle(c)
        self.x_set, self.y_set = zip(*c)

    def load_audio(self,x_fs,return_sample = False):
        '''
        加载音频
        :param x_fs:
        :param return_sample:
        :param vad_cut:是否初步进行端点检测并返回切割后的音频，默认为真
        :return:
        '''
        sampling_freq, audio = wavfile.read(x_fs)

        audio = audio.astype(np.int32)
        if self.vad_cut :
            audio = self.vad.extract_audio(audio)

        if return_sample:
            return  audio,sampling_freq
        else:
            return audio

    def load_label_from_txt(self, y_fs)->np.ndarray:
        '''
        从txt文件中加载拼音，必须保证使用Cleaner中的类依次清洗过数据，保证了格式
        :param y_fs:
        :return:
        '''
        with open(y_fs,encoding="utf-8") as f:
            f.readline()
            line = f.readline().strip()
            pylist = line.split(" ")

            return self.pymap.pylist2vector(pylist)

    def summery(self,audio = True,label = True,plot = False,plot_dir = "./summary",dataset_name = None):
        '''
        统计音频时长、采样率、标签长度、绘制数据的标签、时长分布曲线等
        :return:
        '''
        result = []
        start = time.clock()

        audio_lens = []
        label_lens = []

        if audio:
            max_audio_len = 0
            max_audio_fs = None
            min_audio_len = 160000
            for i,x in enumerate(self.x_set):
                print(f"\rchecked {i} wav files:{x}", end="\0", flush=True)
                audio,sample = self.load_audio(x,True)
                # max_audio_len = max(max_audio_len,audio.shape[0])
                if max_audio_len < audio.shape[0]:
                    max_audio_fs = x
                    max_audio_len = audio.shape[0]
                if plot:
                    audio_lens.append(audio.shape[0])

                min_audio_len = min(min_audio_len,audio.shape[0])

            audio = self.load_audio(max_audio_fs)
            feature = self.melf.mfcc(audio)

            print(f"\nmax audio len = {max_audio_len}, max timestamp = {feature.shape} ,min audio len = {min_audio_len}, sample = {sample}")

        if label:
            py_set = set()
            max_label_len = 0
            min_label_len = 65536
            for i,y in enumerate(self.y_set):
                print(f"\rchecked {i} label files:{y}",end="\0",flush=True)
                label = self.load_label_from_txt(y)
                py_set.update(label.tolist())
                max_label_len = max(max_label_len,label.shape[0])
                min_label_len = min(min_label_len,label.shape[0])
                if plot:
                    label_lens.append(label.shape[0])


            print(f"\nmax label len = {max_label_len}, min label len = {min_label_len}, pinpin coverage:{len(py_set)}")

        end = time.clock()

        print(f"result from {self.set_size} sample, used {end-start} sec")

        if plot:
            os.makedirs(plot_dir,exist_ok=True)

            audio_lens.sort()
            label_lens.sort()

            plt.figure()
            plt.plot(audio_lens)
            plt.ylabel("audio lens")
            plt.xlabel(dataset_name)
            plt.savefig(os.path.join(plot_dir,f"audio_lens_{dataset_name}.png"))
            plt.figure()
            plt.plot(label_lens)
            plt.ylabel("label lens")
            plt.xlabel(dataset_name)

            plt.savefig(os.path.join(plot_dir,f"label_lens_{dataset_name}.png"))

        return result

class TextLoader(DataLoader):
    '''该类用于训练语言模型时候提供语料，但由于时间紧张写的不是很好，接口也没有统一'''
    grain_word="word"
    grain_alpha="alpha"
    def __init__(self,
                 txtfs_set,
                 pinyin_map,
                 chs_map,
                 batch_size=32,
                 padding_length=64,
                 cut_sub=None, create_for_train=True,
                 grain = None,
                 strip_tone = False):
        '''

        :param txtfs_set:
        :param batch_size:
        :param sil_model:
        :param padding_length:
        :param cut_sub:
        :param pinyin_map: 拼音-index 和alpha-index 的字典
            该字典的max_index 一般和编译模型用到的第一层的输入（语言模型）或最后一层的softmax（声学模型）有关
            具体示例可以看example/train_language_model.py 和 train_acoustic_model.py 的具体使用
        :param chs_map:汉字-index 的字典，从外界传入，该字典的max_index一般和编译模型用到的最后一层的softmax的最大值相同
            具体示例可以看example/train_language_model.py 的具体使用
        :param create_for_train:
        :param grain: 粒度，"word"/"alpha"，表示返回的输入的粒度是拼音的index向量还是拼音相应的字母组成的index向量
            如果粒度为"alpha"，那么默认不带声调，a-z分别为1-26，0为padding
        :param strip_tone: 拼音是否去掉声调（即拼音后的12345）
        '''
        super().__init__()
        if cut_sub is not None:
            txtfs_set = txtfs_set[:cut_sub]
        self.txt_fs = txtfs_set

        self.set_size = len(txtfs_set)
        self.chs_map = chs_map
        self.pinyin_map = pinyin_map
        self.batch_size = batch_size
        self.padding_length = padding_length
        self.max_py_size = self.pinyin_map.max_index
        self.max_chs_size = self.chs_map.categores
        self.create_for_train = create_for_train
        if grain is None:
            grain = self.grain_word
        assert grain == self.grain_alpha or grain == self.grain_word,f"param grain must be 'word' or 'alpha',but {grain}"
        self.grain = grain
        self.strip_tone = strip_tone

    def remove_data(self,i):
        '''用于在遇到不在字典里的汉字时，去掉该语料不再将其训练'''
        self.txt_fs.pop(i)
        self.set_size = len(self.txt_fs)

    def get_item(self,i,batch_size = None,mode = "train"):
        if batch_size is None:
            batch_size = self.batch_size

        if mode == TextLoader.train_mode:
            pass

        xs = []
        ys = []

        while True:
            if i > self.set_size-1:
                i = 0

            line,pyline = self.load_sample(self.txt_fs[i])

            if len(line) != len(pyline):
                self.remove_data(i)
                continue

            xs.append(line)
            ys.append(pyline)

            if len(xs)==batch_size:
                break

            i+=1

        label_len = [x.shape[0] for x in xs]

        xs = TextLoader.corpus2feature(xs,self.padding_length)
        ys = TextLoader.corpus2feature(ys,self.padding_length,2,self.max_chs_size)

        label_len = np.array(label_len)
        placeholder = np.zeros_like(label_len)

        return xs,ys

    @staticmethod
    def corpus2feature(xs, feature_pad_len, n_dim = 1, n2size = 5000):
        assert n_dim == 1 or n_dim == 2,"n_dim must be 1 or 2, means [batch,indexs]/[batch,timestame,one-hot]"
        xs = pad_sequences(xs,feature_pad_len,padding="post",truncating="post")
        if n_dim == 2:
            xs = to_categorical(xs,n2size)

        return xs

    def load_sample(self,fs):
        pyline,line = None,None
        with open(fs,encoding="utf-8") as f:
            line = f.readline().strip()
            if self.create_for_train:
                pyline = f.readline().strip()

                pyline = pyline.split(" ")
                if self.strip_tone:
                    pyline = [i.strip("12345") for i in pyline]
                if self.grain == self.grain_alpha:
                    pyline = [i.strip("12345") for i in pyline] # 转换成字母粒度时自动先去掉音调
                    pyline = list("".join(pyline))
                    pyline = [i for i in pyline if len(i) != 0]
                    pyline = self.pinyin_map.alist2vector(pyline) # np.ndarray, (sample,)
                else:
                    pyline = [i for i in pyline if len(i) != 0]
                    pyline = self.pinyin_map.pylist2vector(pyline) # np.ndarray, (sample,)

        line = [i for i in line if i not in filter_char]
        line = self.chs_map.chsent2vector(line)

        return pyline,line

    def __len__(self):
        return self.set_size

    def on_epoch_end(self):
        random.shuffle(self.txt_fs)

    def choice(self):
        index = random.randint(0, self.set_size)
        return self.get_item(index)


class TextLoader2(DataLoader):
    '''第二种提供语料的类，由于一个文件一句话一个拼音对一些大语料需要文件太大了，因此该类提供的读取
        文件a:
            句子\t拼音
        文件b:
            句子\t拼音
    '''
    grain_word="word"
    grain_alpha="alpha"
    def __init__(self,
                 txtfs_set,
                 pinyin_map,
                 chs_map,
                 batch_size=32,
                 padding_length=64,
                 cut_sub=None, create_for_train=True,
                 grain = None,
                 strip_tone = False):
        '''

        :param txtfs_set:
        :param batch_size:
        :param sil_model:
        :param padding_length:
        :param cut_sub:
        :param pinyin_map: 拼音-index 和alpha-index 的字典
            该字典的max_index 一般和编译模型用到的第一层的输入（语言模型）或最后一层的softmax（声学模型）有关
            具体示例可以看example/train_language_model.py 和 train_acoustic_model.py 的具体使用
        :param chs_map:汉字-index 的字典，从外界传入，该字典的max_index一般和编译模型用到的最后一层的softmax的最大值相同
            具体示例可以看example/train_language_model.py 的具体使用
        :param create_for_train:
        :param grain: 粒度，"word"/"alpha"，表示返回的输入的粒度是拼音的index向量还是拼音相应的字母组成的index向量
            如果粒度为"alpha"，那么默认不带声调，a-z分别为1-26，0为padding
        :param strip_tone: 拼音是否去掉声调（即拼音后的12345）
        '''
        super().__init__()
        if cut_sub is not None:
            txtfs_set = txtfs_set[:cut_sub]
        self.txt_fs = txtfs_set

        self.set_size = len(txtfs_set)
        self.chs_map = chs_map
        self.pinyin_map = pinyin_map
        self.batch_size = batch_size
        self.padding_length = padding_length
        self.max_py_size = self.pinyin_map.max_index
        self.max_chs_size = self.chs_map.categores
        self.create_for_train = create_for_train
        if grain is None:
            grain = self.grain_word
        assert grain == self.grain_alpha or grain == self.grain_word,f"param grain must be 'word' or 'alpha',but {grain}"
        self.grain = grain
        if grain == TextLoader2.grain_alpha:
            strip_tone = True
        self.strip_tone = strip_tone
        self._initial()

    def remove_data(self,i):
        '''用于在遇到不在字典里的汉字时，去掉该语料不再将其训练'''
        self.txt_fs.pop(i)
        self.set_size = len(self.txt_fs)

    def _initial(self):
        self.streams = [open(fs,encoding="utf-8") for fs in self.txt_fs]
    def _reload(self,i):
        self.streams[i].close()
        self.streams[i] = open(self.txt_fs[i],encoding="utf-8")

    def get_item(self,i,batch_size = None,mode = "train"):
        if batch_size is None:
            batch_size = self.batch_size

        if mode == TextLoader.train_mode:
            pass

        xs = []
        ys = []

        while True:
            if i > self.set_size-1:
                i = 0

            line,pyline = self.load_line(self.streams[i])
            if line is None:
                self._reload(i)
                continue

            if len(line) != len(pyline):
                continue

            xs.append(line)
            ys.append(pyline)

            if len(xs)==batch_size:
                break

            i+=1

        label_len = [x.shape[0] for x in xs]

        xs = TextLoader.corpus2feature(xs,self.padding_length)
        ys = TextLoader.corpus2feature(ys,self.padding_length,2,self.max_chs_size)

        label_len = np.array(label_len)
        placeholder = np.zeros_like(label_len)

        return xs,ys

    @staticmethod
    def corpus2feature(xs, feature_pad_len, n_dim = 1, n2size = 5000):
        assert n_dim == 1 or n_dim == 2,"n_dim must be 1 or 2, means [batch,indexs]/[batch,timestame,one-hot]"
        xs = pad_sequences(xs,feature_pad_len,padding="post",truncating="post")
        if n_dim == 2:
            xs = to_categorical(xs,n2size)

        return xs

    def _pad_udl(self,word,py):
        word = word+(len(py)-len(word))*"_"
        return word

    def load_line(self, fstream):
        fline = fstream.readline().strip()
        if len(fline) == 0:
            return None,None

        line,pyline = fline.split("\t")
        line = line.strip()
        pyline = pyline.strip().split(" ")

        line = [i for i in line if i not in filter_char]

        if self.strip_tone:
            pyline = [i.strip("12345") for i in pyline]
            # print(pyline)
        if self.grain == self.grain_alpha:
            line = "".join([self._pad_udl(word,py) for word,py in zip(line,pyline)])

        line = self.chs_map.chsent2vector(line)

        if self.grain == self.grain_alpha:
            pyline = list("".join(pyline))
            pyline = [i for i in pyline if len(i) != 0]
            pyline = self.pinyin_map.alist2vector(pyline)  # np.ndarray, (sample,)
        else:
            pyline = [i for i in pyline if len(i) != 0]
            pyline = self.pinyin_map.pylist2vector(pyline)  # np.ndarray, (sample,)

        return pyline, line


    def __len__(self):
        return self.set_size

    def on_epoch_end(self):
        random.shuffle(self.txt_fs)

    def choice(self):
        index = random.randint(0, self.set_size)
        return self.get_item(index)
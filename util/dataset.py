import os
import config
from util.mapmap import PinyinMapper,ChsMapper,StopwordUtil

class Datautil:
    @staticmethod
    def count_label(line):
        '''统计字频汉字要求字符串，拼音要求list'''
        res = {}
        for i in line:
            num = res.setdefault(i,0)
            res[i] = num+1

        return res

    @staticmethod
    def merge_count_label(res:dict,all_res = None):
        if all_res is None:
            all_res = {}
        for k,v in res.items():
            num = all_res.setdefault(k,0)
            all_res[k] = num+v

        return all_res

    @staticmethod
    def write_count_result(path,chs_dict:dict,py_dict:dict,min_count = None):
        chs_dict_path = os.path.join(path,"chs_dict.dict")
        py_dict_path = os.path.join(path,"py_dict.dict")

        with open(chs_dict_path,"w",encoding="utf-8") as w:
            for k,v in chs_dict.items():
                w.write(f"{k}:::{v}\n")
        print(f"write:{chs_dict_path}")

        with open(py_dict_path,"w",encoding="utf-8") as w:
            for k,v in py_dict.items():
                w.write(f"{k}:::{v}\n")
        print(f"write:{py_dict_path}")

        print(f"[info*] Write py and chs dict in {path}.")

    @staticmethod
    def load_dict(path):
        res = {}
        with open(path,encoding="utf-8") as f:
            for line in f:
                k,v = line.strip().split(":::")
                res[k] = int(v)
        return res


    @staticmethod
    def merge_dict(dir_pathlist,output_dir_path = None):
        '''合并生成的数据集的字典'''
        all_chs_res = {}
        all_py_res = {}
        for dir_path in dir_pathlist:
            chs_dict_path = os.path.join(dir_path,"chs_dict.dict")
            py_dict_path = os.path.join(dir_path,"py_dict.dict")

            chs_dict = Datautil.load_dict(chs_dict_path)
            py_dict = Datautil.load_dict(py_dict_path)

            all_chs_res = Datautil.merge_count_label(chs_dict,all_chs_res)
            all_py_res = Datautil.merge_count_label(py_dict,all_py_res)

        Datautil.write_count_result(output_dir_path,all_chs_res,all_py_res)


    @staticmethod
    def filter_dict(dict_path,output_dict_path,min_count = 50):
        with open(dict_path,encoding="utf-8") as f,\
                open(output_dict_path,"w",encoding="utf-8") as w:
            for line in f:
                k,v = line.strip().split(":::")
                v = int(v)
                if v < min_count:
                    continue
                w.write(f"{k}:::{v}\n")
        print(f"write {output_dict_path}")

    @staticmethod
    def check_remove(path):
        if os.path.exists(path):
            os.remove(path)

class Dataset:
    label_mode = "label" #用于生成正确的标签
    clean_mode = "clean" #用于在生成正确的标签后进行清洗
    train_mode = "train" #用于清洗后提供数据集
    ''''''
    def __init__(self,path):
        self.path = path
        self._check()

    def _check(self):
        '''清洗完后，在根目录下生成一个文件，表示无需再清洗了'''
        symbol = os.path.join(self.path,"symbol")
        self.check = os.path.exists(symbol)
        self.pymap = PinyinMapper()
        self.chsmap = ChsMapper()

    def _pre_process_line(self,line):
        '''
        TODO 处理中文字符串
        :param line:str
        :return: 如果存在字母、数字，则返回None
                如果存在标点符号、空格，返回去掉之后的字符串
                不考虑汉字是否在字典中
        '''
    def _pre_process_pyline(self,pyline):
        '''
        TODO 处理拼音字符串
        :param pyline:
        :return: 将所有多余的空格去掉，确保拼音之间只有一个空格
                    如果拼音不在字典中，则返回None
        '''

    def initial(self):
        self.label_dataset()
        self.count_dataset()
        # self.clean_dataset()

    def clean(self):
        self.clean_dataset()

    def label_dataset(self):
        '''对数据生成需要的标签，均为在目录下为wav文件生成相应的txt（清华除外，为wav.trn）文件
                如果出现错误无法生成数据（即无法同时具备音频和文本），则删除相应的wav文件或标签文件

            注意：此时，不保证汉字和拼音一一对应，汉字中可能存在一些无法被注音的数字、字母、标点符号。
        '''
        print(f"[info*]Create labels in {self.__class__.__name__}.")

        dataiter = self.create_fs_iter(mode=Dataset.label_mode)
        for i,(wav_fn,txt_fn,[line,pyline]) in enumerate(dataiter):
            print(f"\r[info*]Process {i},fn = {txt_fn}",end="\0",flush=True)

            if not os.path.exists(wav_fn) and os.path.exists(txt_fn):
                # os.remove(txt_fn)
                print(f"\n{txt_fn} may not have the wav file {wav_fn}, please check it.")
                continue
            if line is None and os.path.exists(wav_fn): # 没有中文但是有wav文件
                # os.remove(wav_fn)
                print(f"\n{wav_fn} not have the labels, it will be deleted.")
                continue

            if pyline is None or len(pyline) == 0: # 没有拼音的话
                pyline = self.pymap.sent2pylist(line)  # 转化为拼音
            else:
                continue # 目前只有清华的数据集全都有，所以不用清洗，不用重写
            with open(txt_fn,"w",encoding="utf-8") as w:
                w.write(f"{line}\n")
                w.write(f"{pyline}\n")
        print()

    def count_dataset(self):
        '''统计两个词典到数据的根目录下，具体如何整合需要使用者自行整理'''
        '''加载训练测试用数据集，使用train_mode'''
        print(f"[info*]Create dicts in {self.__class__.__name__}.")
        dataiter = self.create_fs_iter(mode=Dataset.train_mode)

        chs_all_dict = {}
        py_all_dict = {}
        for _,txt_fn in dataiter:
            with open(txt_fn,encoding="utf-8") as f:
                line = f.readline().strip()
                pyline = f.readline().strip().split(" ")
                pyline = [i.strip("5\n") for i in pyline]

            chs_dict = Datautil.count_label(line)
            py_dict = Datautil.count_label(pyline)
            chs_all_dict = Datautil.merge_count_label(chs_dict,chs_all_dict)
            py_all_dict = Datautil.merge_count_label(py_dict,py_all_dict)

        Datautil.write_count_result(path=self.path,
                                    chs_dict=chs_all_dict,
                                    py_dict=py_all_dict)


    def clean_dataset(self):
        '''根据最终确定的词典结果进行清理
        '''
        dataiter = self.create_fs_iter(mode=Dataset.train_mode) # 清洗时候需要的数据格式和train_mode一样
        count = 0
        oov_count = 0
        for i,(wav_fn,txt_fn) in enumerate(dataiter):
            print(f"\r{i},err_count = {count},oov_count = {oov_count},fn = {txt_fn[:-20]}",end="\0",flush=True)
            with open(txt_fn,encoding="utf-8") as f:
                line = f.readline().replace(" ","").replace("　","").strip()
                pyline = f.readline().strip().split(" ")
            new_line = StopwordUtil.clean_line(line)
            if new_line is None:
                Datautil.check_remove(wav_fn)
                Datautil.check_remove(txt_fn)
                count+=1
            elif len(new_line) != len(pyline):
                Datautil.check_remove(wav_fn)
                Datautil.check_remove(txt_fn)
                count+=1

            no_oov,oov_dict = self.pymap.check_line(pyline)
            if not no_oov:
                oov_count+=1
                Datautil.check_remove(wav_fn)
                Datautil.check_remove(txt_fn)

        print()


    def load_dataset(self):
        '''加载训练测试用数据集，使用train_mode'''
        dataiter = self.create_fs_iter(mode=Dataset.train_mode)
        x_set = []
        y_set = []
        for x,y in dataiter:
            x_set.append(x)
            y_set.append(y)

        return x_set,y_set

    def create_fs_iter(self,mode="train"):
        raise NotImplementedError(
            f"create_fs_iter() must be Implemented in {self.__class__.__name__}")

class Thchs30(Dataset):
    '''子数据集类下用于获取具体的文本文档和wav文件，用迭代器的方式向基类中提供：
            wavfs,[line，pyline]
        父类如果要提供训练用数据集，则只接受wavf和txtf
            如果要生成标签，则接受wavf,txtf,[line,pyline]，其中line和pyline重写到txtf中去
    '''
    def create_fs_iter(self,mode = "train"):
        path = os.path.abspath(self.path)
        datapath = os.path.join(path,"data") # ./data/

        fs = os.listdir(datapath) # .wav / .trn

        fs = [os.path.join(datapath,i) for i in fs if i.endswith(".wav")]
        for wav_fn in fs:
            txt_fn = f"{wav_fn}.trn"
            if mode == Dataset.train_mode:
                yield wav_fn,txt_fn
            elif mode == Dataset.label_mode:
                with open(txt_fn,encoding="utf-8") as f:
                    line = f.readline().strip()
                    pyline = f.readline().strip().split(" ")
                yield wav_fn,txt_fn,[line,pyline]




class Z200(Dataset):
    def create_fs_iter(self,mode = "train"):
        path = os.path.abspath(self.path)
        root = os.listdir(path)
        root = [os.path.join(path, i) for i in root]
        root = [os.path.join(i, "session01") for i in root if os.path.isdir(i)]
        fs = []

        for sub_dir in root:
            sub_fs = os.listdir(sub_dir)
            sub_fs = [os.path.join(sub_dir, i[:-4]) for i in sub_fs if i.endswith(".txt")]
            # fs.extend(sub_fs)
            for fn in sub_fs:
                wav_fn = f"{fn}.wav"
                txt_fn = f"{fn}.txt"
                if mode == Dataset.train_mode:
                    yield wav_fn,txt_fn
                elif mode == Dataset.label_mode:
                    with open(txt_fn,encoding="utf-8") as f:
                        line = f.readline().strip()
                        pyline = f.readline().strip()
                    yield wav_fn,txt_fn,[line,pyline]

class Primewords(Dataset):
    def create_fs_iter(self,mode = "train"):
        import json
        json_path = os.path.join(self.path,"set1_transcript.json")
        assert os.path.exists(json_path),"set1_transcript.json not exists!"

        file_label_map = {}
        if mode == Dataset.label_mode:
            with open(json_path,encoding="utf-8") as f:
                jdict = json.load(f)
            for sample in jdict:
                file_label_map[sample["file"]] = sample["text"] # 格式 ： "632b4316-d806-448a-b549-650d7e233b80.wav" : "总共有三十支球队 分为东部联盟 西部联盟"


        '''获取每个wav文件'''
        audio_root_dir = os.path.join(self.path,"audio_files") # ./audio_files/
        f0 = os.listdir(audio_root_dir) # 从0-f 的16个目录
        f0 = [os.path.join(audio_root_dir,f) for f in f0]

        ff00 = []  # 包含所有00-ff的文件夹路径
        for f in f0:
            subff = os.listdir(f) # 列出00-ff几个文件夹
            subff = [os.path.join(f,sf) for sf in subff]
            ff00.extend(subff)

        for subff in ff00:
            wavfs = os.listdir(subff)
            wavfs = [wavf for wavf in wavfs if wavf.endswith(".wav")] # 过滤审查一遍
            for wav_fn in wavfs:
                fpre, _ = os.path.splitext(wav_fn)
                key = wav_fn

                wav_fn = os.path.join(subff,f"{fpre}.wav")
                txt_fn = os.path.join(subff,f"{fpre}.txt")

                if mode == Dataset.label_mode:
                    line = file_label_map.get(key, None)
                    yield wav_fn,txt_fn,[line,None] # 说明这里的line有可能为None

                elif mode == Dataset.train_mode:
                    yield wav_fn,txt_fn

class AiShell(Dataset):
    def create_fs_iter(self,mode="train"):
        file_label_map = {}
        if mode == Dataset.label_mode:
            label_file = os.path.join(self.path, "transcript/aishell_transcript_v0.8.txt")
            assert os.path.exists(
                label_file), "file 'aishell_transcript_v0.8.txt' not exists, please check dir ./transcript/ ! "

            with open(label_file,encoding="utf-8") as f:
                for line in f:
                    file,label = line.split(" ",maxsplit=1)

                    fpre,_ = os.path.splitext(file)
                    file_label_map[fpre] = label.strip()

        train_root = os.path.join(self.path,"wav/train")
        test_root = os.path.join(self.path,"wav/test")
        dev_root = os.path.join(self.path,"wav/dev")

        for fs in [train_root,test_root,dev_root]: #每一个目录
            fs = self._get_sub_wavs(fs)
            for wav_fn in fs: # 每一个wav文件
                path, fname = os.path.split(wav_fn)
                fpre,ext = os.path.splitext(fname)

                txt_fn = f"{fpre}.txt"
                txt_fn = os.path.join(path,txt_fn)

                if mode == Dataset.train_mode:
                    yield wav_fn,txt_fn
                elif mode == Dataset.label_mode:
                    line = file_label_map.get(fpre, None)
                    yield wav_fn,txt_fn,[line,None] # 注意这里的line可能为None，说明该文件没有被标注，只有wav文件

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

        fs = [f for f in fs if f.endswith(".wav")] # 过滤审查

        return fs

class ST_CMDS(Dataset):
    def create_fs_iter(self,mode="train"):
        fs = os.listdir(self.path)
        fs = [os.path.join(self.path, f[:-4]) for f in fs if f.endswith(".wav")]

        for i, fpre in enumerate(fs):
            wav_fn = f"{fpre}.wav"
            txt_fn = f"{fpre}.txt"
            if mode == Dataset.train_mode:
                yield wav_fn,txt_fn
            elif mode == Dataset.label_mode:
                with open(txt_fn,encoding="utf-8") as f:
                    line = f.readline().strip()
                yield wav_fn,txt_fn,[line,None]
            elif mode == Dataset.clean_mode:
                pass

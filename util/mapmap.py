import os,numpy as np
import config
from pypinyin import pinyin,Style,load_phrases_dict
import re
import json,random


class StopwordUtil():
    re_stop = re.compile("[×“”~@…、。」！（），？=,\"'!().?]")
    re_cant_py = re.compile("[0-9a-zA-ZａＡｂｃｋｔ]")
    @staticmethod
    def exist_cant_py(line):
        ''''''
        return re.search(StopwordUtil.re_cant_py,line) is not None

    @staticmethod
    def sub_cant_stop(line):
        return re.sub(StopwordUtil.re_stop,"",line)

    @staticmethod
    def clean_line(line):
        if not StopwordUtil.exist_cant_py(line):
            return StopwordUtil.sub_cant_stop(line)
        else:
            return None

class ChsMapper():
    def __init__(self, chsfile = None,pad_mode = 0):
        if chsfile is None:
            chsfile = config.chs_dict_path
        self.chsfile = chsfile
        self.pad_mode = pad_mode
        self.load()

    def load(self):
        word_num_map = {}
        num_word_map = {}

        i = 0
        with open(self.chsfile,encoding="utf-8") as f:
            for i,w in enumerate(f):
                w = w.strip()
                if self.pad_mode == 0:
                    word_num_map[w] = i+1
                    num_word_map[i+1] = w

        word_num_map[""] = 0
        num_word_map[0] = "_"
        self.max_index = i+1
        self.categores = i+2
        self.word_num_map = word_num_map
        self.num_word_map = num_word_map

    def word2num(self,x):
        x = x.strip("_")
        res = self.word_num_map.get(x,None)
        # if res is None:
        #     print(x)
        return res

    def num2word(self,x):
        return self.num_word_map.get(x,None)

    def chsent2vector(self,sample):
        sample = [self.word2num(x) for x in sample]
        sample = [i for i in sample if i is not None]
        sample = np.array(sample)
        return sample

    def vector2chsent(self,sample,return_seq = False):
        sample = [self.num2word(x) for x in sample]
        if return_seq:
            sample = "".join(sample)
        return sample

    def batch_vector2chsent(self,batch,return_seq = True,return_line = False):
        batch = [self.vector2chsent(sample,return_seq) for sample in batch]
        if return_line:
            batch = " ".join(batch)
        return batch

    def batch_chsent2vector(self,batch):
        batch = [self.chsent2vector(sample) for sample in batch]
        batch = np.array(batch)

        return batch

    def check_line(self,line:str) -> dict:
        oov = {}
        for chs in line:
            if chs not in self.word_num_map:
                i = oov.setdefault(chs,0)
                oov[chs] = i+1
        return oov

class ErrPinyinMapper():
    def __init__(self,errdictfile = None):
        if errdictfile is None:
            errdictfile = config.err_dict_path
        with open(errdictfile,encoding="utf-8") as f:
            self.errdict = json.load(f)

    def replace(self,s):
        if s not in self.errdict:
            return s
        ap = self.errdict[s]
        res = random.choice(ap)
        # print(s,res)
        return res

class PinyinMapper():
    '''
    用于建立拼音映射， 包括拼音和序号之间的，拼音和文字之间的

    拼音到序号的映射有几种
        在开头添加一行为静音
        在结尾添加一行为
    '''
    def __init__(self,pyfile = None,sil_mode = 0):
        '''

        :param pyfile:
        :param use_pinyin: 是否使用pinyin 库注音
        :param sil_mode: 决定静音音素的标识，0代表第一个，1代表最后一个，-1代表考虑ctc解码，结尾一个,-1 1个
        '''
        if pyfile is None:
            pyfile = config.py_dict_path
        self.py_file = pyfile
        self.max_index = None  # 这个max_index 是包含了代码添加的blank后的max_index，num_py_map[max_index] = <blank>
        self.sil_mode = sil_mode

        self.use_pinyin = True #use_pinyin 该选项在之后会废弃，永久使用pinyin库注拼音


        # 这里是发现的一些错误，暂时用硬编码解决了
        change_dict = {
            "茸": [["rong2"]],
            "蓉": [["rong2"]],
            "嗯": [["en1"]],
            "哦": [["ō"]],
            "排场": [["rong2"], ["chang3"]],
            "难进易退": [["nan2"], ["jin4"], ["yi4"], ["tui4"]],
            "哭丧": [["ku1"], ["sang4"]],

        }
        load_phrases_dict(change_dict, )

        self.pinyin = lambda word:pinyin(word,Style.TONE3,errors="ignore")

        self.load()

    def summary(self):
        print(self.py_num_map)
        print(self.num_py_map)

    def load(self):
        py_word_map = {}  # 存储了字：拼音的字典列表
        word_py_map = {}
        py_num_map = {}  # 存储了拼音:index 的字典，用于创建npy
        num_py_map = {}
        with open(self.py_file, encoding="utf-8") as f:
            index = 0
            for index, line in enumerate(f):
                py = line.strip()
                # for i in words:
                #     lst = py_word_map.setdefault(i, [])
                #     lst.append(py)
                #     word_py_map[i] = py

                if self.sil_mode == 0: #因为第0号是静音音素，所以加1
                    py_num_map[py] = index+1
                    num_py_map[index+1] = py
                elif self.sil_mode == 1: #其他模式不影响，正常排列
                    py_num_map[py] = index
                    num_py_map[index] = py
                elif self.sil_mode == -1:
                    py_num_map[py] = index
                    num_py_map[index] = py


            print(f"Load pinyin dict. Max index = {index if self.sil_mode != 0 else index+1}.")

            if self.sil_mode == 1: #放到最后一个
                py_num_map["-"] = index+1
                num_py_map[index+1] = "-"
                self.max_index = index + 1
            elif self.sil_mode == 0: #放到第一个
                py_num_map["-"] = 0
                num_py_map[0] = "-"
                self.max_index = index + 1
            elif self.sil_mode == -1: #设置为-1
                py_num_map["-"] = 0
                # num_py_map[0] = "-"
                num_py_map[-1] = "" #
                num_py_map[index+1] = "-"
                self.max_index = index + 1

        self.py_word_map= py_word_map
        self.word_py_map = word_py_map
        self.py_num_map = py_num_map
        self.num_py_map = num_py_map

        alpha = "_abcdefghijklmnopqrstuvwxyz"
        self.alpha_num_map = {a:i for i, a in enumerate(alpha)}
        self.num_alpha_map = {i:a for i, a in enumerate(alpha)}


    def num2alpha(self,x):
        res = self.num_alpha_map.get(x,None)
        return res

    def alpha2num(self,x):
        res = self.alpha_num_map.get(x,None)
        # if res is None:
        #     print(x)
        return res

    def alist2vector(self, sample):
        sample = [self.alpha2num(x) for x in sample]
        sample = np.array(sample)
        return sample

    def vector2alist(self,sample):
        sample = [self.num2alpha(x) for x in sample]
        sample = "".join(sample)
        return sample

    def batch_alist2vector(self,batch):
        batch = [self.alist2vector(sample) for sample in batch]
        batch = np.array(batch)
        return batch

    def batch_vector2alist(self,batch):
        batch = [self.vector2alist(sample) for sample in batch]
        return batch

    def num2py(self,x):
        res = self.num_py_map.get(x,None)
        return res

    def py2num(self,x:str):
        x = x.strip("5").strip() #因为字典里轻声没有5，所以手动去掉，一视同仁
        res = self.py_num_map.get(x,None)
        if res is None:
            print(f"\n'{x}' is None,please check dict.")
        return res

    def sent2pylist(self,sample:str,to_str = True)->[str,list]:
        '''
        :param sample: str,可能包含空格，或者不包含空格
        :return:
        '''
        result = self.pinyin(sample)
        result = [py[0] for py in result if " " not in py[0]]
        if to_str:
            return " ".join(result)
        else:
            return result

    def pylist2vector(self,sample):
        sample = [self.py2num(x) for x in sample if x is not None]
        sample = np.array(sample)
        return sample

    def vector2pylist(self,sample,return_word_list = False):
        sample = [self.num2py(x) for x in sample]
        sample = " ".join(sample).strip()
        sample = sample.replace("- ","")
        if return_word_list:
            return sample.split(" ")

        return sample

    def batch_vector2pylist(self,batch,return_word_list = False,return_list = False):
        batch = [self.vector2pylist(sample,return_word_list = return_word_list) for sample in batch]
        if return_list:
            return batch
        return "\n".join(batch)

    def check_line(self,pyline:list):
        oov = {}
        for py in pyline:
            py = py.strip("5\n")
            if py not in self.py_num_map:
                i = oov.setdefault(py,0)
                oov[py] = i+1
        return len(oov) == 0,oov


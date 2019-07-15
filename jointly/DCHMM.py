from core.base_model import BaseJoint
import numpy as np
from util.mapmap import PinyinMapper,ChsMapper
from language.HMM import PHHMM
from acoustic.ABCDNN import DCBNN1D

class DCHMM(BaseJoint):
    '''DCBNN1D + HMM ，HMM模型来自于库 Pinyin2Hanzi:https://github.com/letiantian/Pinyin2Hanzi'''
    def compile(self, ac_model_load_path):
        model_helper = DCBNN1D(self.py_map)
        model_helper.compile(feature_shape=self.acmodel_input_shape,
                             label_max_string_length=64, #该长度用于提供训练，测试时无需使用
                             ms_output_size=self.py_map.max_index+1)
        self.ac_model = model_helper


        self.py_w = 64
        model_helper = PHHMM()
        self.lg_model = model_helper

        if ac_model_load_path is not None:
            self.ac_model.load(ac_model_load_path)

    def voice_reco(self,xs):
        '''
        返回py_index_list
        :param xs:
        :return:
        '''
        xs, feature_len = self.pre_process_audio(xs)

        batch = [xs, None, feature_len, None], None
        prob_result = self.ac_model.prob_predict(batch)

        argmax_res,prob = self.ac_model.ctc_decoder.ctc_decode(prob_result, feature_len,return_prob=True)
        pylist_pred = self.py_map.batch_vector2pylist(argmax_res, return_word_list=True, return_list=True)

        argmax_res[argmax_res == -1] = 0

        return argmax_res,pylist_pred,prob

    def raw_record(self, xs):
        '''最底层的识别方法，可以传入音频，由控制台调用（这两个都已经在该类里写好），也可以由UI调用
        每个类的实现不同，需要自行实现
        '''
        xs = self.audio_tool.noise_filter(xs)
        py_index_list,pylist,ctc_prob = self.voice_reco(xs)

        # 该行代码用于padding pyindex，用于输入语言模型（不一定需要输入，使用传统语言模型时只需要pylist即可）
        # py_pad = TextLoader.corpus2feature(py_index_list,self.py_w)

        pyline = np.concatenate(pylist).tolist()

        chlist,prob = self.lg_model.predict(pyline)

        chline = " ".join(chlist)

        '''TODO 未规范返回值，慎用'''
        return pyline,chline,[prob]

    @staticmethod
    def real_predict(path="./model/DCBNN1D_cur_best.h5"):
        '''
        :param path:DCBNN1D的预训练权重文件路径
        :return:
        '''
        dcnn = DCHMM(
            acmodel_input_shape=(1600, 200),
            acmodel_output_shape=(200,),
            lgmodel_input_shape=None,
            py_map=PinyinMapper(sil_mode=-1),
            chs_map=ChsMapper())

        dcnn.compile(path)

        while True:
            pyline, chline, prob = dcnn.record_from_cmd(3)
            print(pyline, chline, prob)



if __name__ == "__main__":
    dcnn = DCHMM(
        acmodel_input_shape=(1600,200),
        acmodel_output_shape=(200,),
        lgmodel_input_shape=None,
        py_map=PinyinMapper(sil_mode=-1),
        chs_map=ChsMapper())

    dcnn.compile("../model/DCBNN1D_cur_best.h5")

    while True:
        pyline, chline, prob = dcnn.record_from_cmd(3)
        print(pyline,chline,prob)
    # dcnn.raw_record()
    # dcnn.run("../model/DCBNN1Dplus_cur_best.h5")
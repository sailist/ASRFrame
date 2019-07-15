from util.mapmap import PinyinMapper,ChsMapper
import numpy as np
from util.reader import TextLoader,TextLoader2
from acoustic.ABCDNN import DCBNN1D
from language.SOMM import SOMMalpha
from core.base_model import BaseJoint

class DCSOM(BaseJoint):
    '''DCBNN1D + SOMMalpha ，目前效果最好'''
    def compile_acmodel(self,ac_model_path):
        assert ac_model_path is not None,"The pre-trained model path must not be None."

        model_helper = DCBNN1D(self.py_map)
        model_helper.compile(feature_shape=self.acmodel_input_shape,
                             label_max_string_length=64,  # 该长度用于提供训练，测试时无需使用
                             ms_output_size=self.py_map.max_index + 1)
        model_helper.load(ac_model_path)
        self.ac_model = model_helper

    def compile_lgmodel(self,lg_model_path):

        model_helper = SOMMalpha()
        model_helper.compile(feature_shape=self.lgmodel_input_shape,
                             ms_pinyin_size=self.py_map.max_index,
                             ms_output_size=self.chs_map.categores)
        model_helper.load(lg_model_path)
        self.lg_model = model_helper

    def compile(self,ac_model_path,lg_model_path):
        self.compile_acmodel(ac_model_path)
        self.compile_lgmodel(lg_model_path)

    def voice_reco(self,xs):
        '''
        返回py_index_list
        :param xs:
        :return:
        '''
        xs, feature_len = self.pre_process_audio(xs)

        batch = [xs, None, feature_len, None], None
        prob_result = self.ac_model.prob_predict(batch)

        argmax_res,ctc_prob = self.ac_model.ctc_decoder.ctc_decode(prob_result, feature_len,return_prob=True)
        pylist_pred = self.py_map.batch_vector2pylist(argmax_res, return_word_list=True, return_list=True)

        argmax_res[argmax_res == -1] = 0

        return argmax_res,pylist_pred,ctc_prob


    def raw_record(self, xs):
        xs = self.audio_tool.noise_filter(xs)
        py_index_list_batch, pylist_batch,ctc_prob = self.voice_reco(xs)

        raw_pylist_batch = [[i.strip("12345") for i in sample] for sample in pylist_batch]
        alpha_batch = ["".join(sample) for sample in raw_pylist_batch]

        alpha_vector_batch = self.py_map.batch_alist2vector(alpha_batch)
        alpha_vector_batch = TextLoader2.corpus2feature(alpha_vector_batch,self.lgmodel_input_shape[0])

        ch_list_batch,prob_batch = self.lg_model.predict([alpha_vector_batch,None],True)

        pyline = np.concatenate(pylist_batch).tolist()
        chline = ",".join(ch_list_batch).replace("_","")

        print(pyline,chline)
        return pyline,chline,[ctc_prob[0]]

    @staticmethod
    def real_predict(ac_path="./model/DCBNN1D_cur_best.h5", lg_path="./model/language/SOMMalpha_step_18000.h5"):
        dcs = DCSOM(acmodel_input_shape=(1600, 200),
                    acmodel_output_shape=(200,),
                    lgmodel_input_shape=(200,),
                    py_map=PinyinMapper(sil_mode=-1),
                    chs_map=ChsMapper(),
                    divide_feature=8)

        dcs.compile(ac_path, lg_path)
        while True:
            try:
                print(dcs.record_from_cmd(5))
            except:
                print("[info*]未识别到语音")
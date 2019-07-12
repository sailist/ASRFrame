# 请在根目录下运行，不然可能会找不到模型
import os
from keras.preprocessing.sequence import pad_sequences
from util.mapmap import PinyinMapper,ChsMapper
from jointly.DCHMM import DCHMM
from jointly.DCSOM import DCSOM
from language.SOMM import SOMMalpha

dir_path = os.path.split(os.path.realpath(__file__))[0] #"./util"

def predict_dchmm(path = "./model/DCBNN1D_cur_best.h5"):
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

def predict_dcsom(ac_path = "./model/DCBNN1D_cur_best.h5",lg_path = "./model/language/SOMMalpha_step_18000.h5"):
    dcs = DCSOM(acmodel_input_shape=(1600,200),
                acmodel_output_shape=(200,),
                lgmodel_input_shape=(200,),
                py_map=PinyinMapper(sil_mode=-1),
                chs_map=ChsMapper(),
                divide_feature=8)

    dcs.compile(ac_path,lg_path)
    while True:
        try:
            print(dcs.record_from_cmd(10))
        except:
            print("[info*]未识别到语音")

def predict_sommalpha(path):
    max_label_len = 200
    pinyin_map = PinyinMapper(sil_mode=0)
    chs_map = ChsMapper()

    model_helper = SOMMalpha()
    model_helper.compile(feature_shape=(max_label_len,),
                         ms_pinyin_size=pinyin_map.max_index,
                         ms_output_size=chs_map.categores)

    model_helper.load(path)

    while True:
        string = input("请输入拼音:")
        xs = [pinyin_map.alist2vector(string)]
        print(xs)
        batch = pad_sequences(xs,maxlen=max_label_len,padding="post",truncating="post"),None
        result = model_helper.predict(batch)
        print(result)


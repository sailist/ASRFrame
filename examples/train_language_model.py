from util.reader import *
from util.mapmap import PinyinMapper,ChsMapper
from language.DCNN import DCNN1D
from language.SOMM import SOMMword,SOMMalpha


def train_dcnn1d(datagene:list,load_model = None):

    dataset = VoiceDatasetList()
    _,y_set = dataset.merge_load(datagene,choose_x=False,choose_y=True)

    max_label_len = 64
    pinyin_map = PinyinMapper(sil_mode=0)
    chs_map = ChsMapper()
    tloader = TextLoader(y_set,padding_length = max_label_len,pinyin_map=pinyin_map,cut_sub=16,
                         chs_map=chs_map)

    model_helper = DCNN1D()
    model_helper.compile(feature_shape=(max_label_len,tloader.max_py_size),
                         ms_input_size=pinyin_map.max_index,
                         ms_output_size=chs_map.categores)

    if load_model is not None:
        model_helper.load(load_model)

    model_helper.fit(tloader,-1)


def train_somiao(datagene:list,load_model = None):

    dataset = VoiceDatasetList()
    _,y_set = dataset.merge_load(datagene,choose_x=False,choose_y=True)

    max_label_len = 64

    pinyin_map = PinyinMapper(sil_mode=0)
    chs_map = ChsMapper()

    tloader = TextLoader(y_set,padding_length = max_label_len,pinyin_map=pinyin_map,chs_map=chs_map)

    model_helper = SOMMword()
    model_helper.compile(feature_shape=(max_label_len,),
                         ms_pinyin_size=pinyin_map.max_index,
                         ms_output_size=chs_map.categores)

    if load_model is not None:
        model_helper.load(load_model)

    model_helper.fit(tloader,-1)

def train_sommalpha(datagene:TextDataGenerator, load_model = None):

    txtfs = datagene.load_from_path()

    max_label_len = 200

    pinyin_map = PinyinMapper(sil_mode=0)
    chs_map = ChsMapper()

    tloader = TextLoader2(txtfs,padding_length = max_label_len,pinyin_map=pinyin_map,chs_map=chs_map,
                          grain=TextLoader2.grain_alpha,
                          cut_sub=175,
                          )

    model_helper = SOMMalpha()
    model_helper.compile(feature_shape=(max_label_len,),
                         ms_pinyin_size=pinyin_map.max_index,
                         ms_output_size=chs_map.categores)

    if load_model is not None:
        model_helper.load(load_model)

    model_helper.fit(tloader,-1)
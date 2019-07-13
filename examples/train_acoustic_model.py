from acoustic import *
from acoustic import DCNN2D
from acoustic.MAXM import MCONM
from acoustic.WAVE import WAVEM
from util.reader import *
from feature.mel_feature import MelFeature5
from util.mapmap import PinyinMapper

def train_templete(datagenes:list,load_model=None):
    '''
    训练方法的脚本模板，有详细的注释
    :param datagenes: Datagene类的list，会由DatasetList加载，合并生成路径
    :param load_model: 如果模型要加载模型的话，这里提供路径
    '''
    w, h = 1600, 200 # 模型的input_shape
    dataset = VoiceDatasetList() # 标准写法
    x_set, y_set = dataset.merge_load(datagenes) #直接传入，可以得到相应的数据集路径

    # 因为sil_mode不同字典映射可能会有不同，因此Voiceloader/模型共用一个pymap，保证不会出现错误
    pymap = PinyinMapper(sil_mode=-1)

    # 生成迭代器，最简单的写法，但如果这么写多半是跑不通的，需要改一些参数，具体可以看VoiceLoader的初始化方法
    vloader = VoiceLoader(x_set, y_set,pymap)

    # 如果是自己写的模型，需要继承AcousticModel，随后在compile方法中定义自己的模型
    model_helper = AcousticModel(pymap)

    # compile方法是自己写的，一般需要input的shape，特征的最大步长，拼音的最大序号，最大步长等参数
    # 编写时参考一下其他模型的写法，注意接口的调用
    # model_helper.compile()

    if load_model is not None:
        load_model = os.path.abspath(load_model)
        model_helper.load(load_model)

    model_helper.fit(vloader,epoch=-1,use_ctc=True) # 固定写法，具体的含义看fit方法的注释

def train_dcnn2d(datagenes:list, load_model = None):
    w, h = 1600, 200

    dataset = VoiceDatasetList()
    x_set, y_set = dataset.merge_load(datagenes)

    pymap = PinyinMapper(sil_mode=-1)
    vloader = VoiceLoader(x_set, y_set,
                          batch_size= 16,
                          n_mels=h, feature_pad_len=w, feature_dim=3,
                          pymap = pymap,
                          melf=MelFeature5(),
                          divide_feature_len=8,)

    model_helper = DCNN2D(pymap)
    model_helper.compile(feature_shape=(w,h,1),ms_output_size=pymap.max_index+1) # ctcloss 计算要求： index < num_class-1

    if load_model is not None:
        load_model = os.path.abspath(load_model)
        model_helper.load(load_model)

    model_helper.fit(vloader,epoch=-1,use_ctc=True)

def train_dcbnn2d(path, load_model = None):
    w,h = 1600,200

    thu_data = Thchs30(path)
    x_set,y_set = thu_data.load_from_path()

    pymap = PinyinMapper(sil_mode=-1)
    vloader = VoiceLoader(x_set,y_set,batch_size=16,n_mels=h,feature_pad_len=w,feature_dim=3,cut_sub=32)

    model_helper = DCBNN2D(pymap)
    model_helper.compile(feature_shape=(w,h,1),label_max_string_length=32,ms_output_size=1423)

    if load_model is not None:
        load_model = os.path.abspath(load_model)
        model_helper.load(load_model)

    model_helper.fit(vloader)

def train_dcbnn1d(datagenes:list, load_model = None):

    w, h = 1600, 200
    max_label_len = 64

    dataset = VoiceDatasetList()
    x_set, y_set = dataset.merge_load(datagenes)
    pymap = PinyinMapper(sil_mode=-1)
    vloader = VoiceLoader(x_set, y_set,
                          batch_size= 16,
                          feature_pad_len = w,
                          n_mels=h,
                          max_label_len=max_label_len,
                          pymap=pymap,
                          melf=MelFeature5(),
                          divide_feature_len=8,
                          all_train=False,
                          )

    model_helper = DCBNN1D(pymap)
    model_helper.compile(feature_shape=(w, h), label_max_string_length=max_label_len, ms_output_size=pymap.max_index+1)

    if load_model is not None:
        load_model = os.path.abspath(load_model)
        model_helper.load(load_model)

    model_helper.fit(vloader,epoch=-1,save_step=1000,use_ctc=True)


def train_wavenet(datagenes:list,load_model = None):
    w,h = None,200
    max_label_len = 64

    dataset = VoiceDatasetList()
    x_set, y_set = dataset.merge_load(datagenes)
    pymap = PinyinMapper(sil_mode=-1)
    vloader = VoiceLoader(x_set, y_set,
                          batch_size=16,
                          feature_pad_len=w,
                          n_mels=h,
                          max_label_len=max_label_len,
                          pymap=pymap,
                          melf=MelFeature5(),
                          divide_feature_len=8,
                          all_train=False,
                          )

    model_helper = WAVEM(pymap)
    model_helper.compile(feature_shape=(w, h), label_max_string_length=max_label_len,
                         ms_output_size=pymap.max_index + 1)

    if load_model is not None:
        load_model = os.path.abspath(load_model)
        model_helper.load(load_model)

    model_helper.fit(vloader, epoch=-1, save_step=1000, use_ctc=True)

def train_mconm(datagenes:list, load_model = None):

    w, h = 800, 200
    max_label_len = 64

    dataset = VoiceDatasetList()
    x_set, y_set = dataset.merge_load(datagenes)
    pymap = PinyinMapper(sil_mode=-1)
    vloader = VoiceLoader(x_set, y_set,
                          batch_size= 16,
                          feature_pad_len = w,
                          n_mels=h,
                          max_label_len=max_label_len,
                          pymap=pymap,
                          melf=MelFeature5(),
                          all_train=False
                          )

    model_helper = MCONM(pymap)
    model_helper.compile(feature_shape=(w, h), label_max_string_length=max_label_len, ms_output_size=pymap.max_index+1)

    if load_model is not None:
        load_model = os.path.abspath(load_model)
        model_helper.load(load_model)

    model_helper.fit(vloader,epoch=-1,save_step=1000,use_ctc=True)


def train_dcbann1d(datagenes:list, load_model = None):
    w, h = 1600, 200
    max_label_len = 64

    dataset = VoiceDatasetList()
    x_set, y_set = dataset.merge_load(datagenes)
    pymap = PinyinMapper(sil_mode=-1)
    vloader = VoiceLoader(x_set, y_set,
                          batch_size= 16,
                          feature_pad_len = w,
                          n_mels=h,
                          max_label_len=max_label_len,
                          pymap=pymap,
                          melf=MelFeature5(),
                          divide_feature_len=8,

                          )

    model_helper = DCBANN1D(pymap)
    model_helper.compile(feature_shape=(w, h), label_max_string_length=max_label_len, ms_output_size=pymap.max_index+1)

    if load_model is not None:
        load_model = os.path.abspath(load_model)
        model_helper.load(load_model)

    model_helper.fit(vloader,epoch=-1,save_step=1000,use_ctc=True)

def train_dcbnn1dplus(datagenes:list, load_model = None):
    w, h = 1600, 200
    max_label_len = 64
    batch_size = 16

    dataset = VoiceDatasetList()
    x_set, y_set = dataset.merge_load(datagenes)
    pymap = PinyinMapper(sil_mode=-1)
    vloader = VoiceLoader(x_set, y_set,
                          batch_size= batch_size,
                          feature_pad_len = w,
                          n_mels=h,
                          max_label_len=max_label_len,
                          pymap=pymap,
                          melf=MelFeature5(),
                          all_train=True,
                          divide_feature_len=8,)

    model_helper = DCBNN1Dplus(pymap)
    model_helper.compile(feature_shape=(w, h), label_max_string_length=max_label_len, ms_output_size=pymap.max_index+1)

    if load_model is not None:
        load_model = os.path.abspath(load_model)
        model_helper.load(load_model)

    # model_helper.fit(vloader,epoch=-1, save_step=len(x_set)//batch_size, use_ctc=True)
    model_helper.fit(vloader,epoch=-1, save_step=len(x_set)//batch_size//30, use_ctc=True)

# def train_las(path,load_model = None):
#     '''
#     停止维护，2019年6月27日，时间有限没法清楚能否训练出来，以后有机会再测试一下
#         该方法因为类的变更，不保证能够运行成功
#     '''
#     w,h = 1024,128
#
#     thu_data = Thchs30(path)
#     x_set,y_set = thu_data.load_from_path()
#
#     model_helper = LASModel()
#     model_helper.compile(feature_shape=(w,h),ms_output_size=1437)
#     if load_model is not None:
#         load_model = os.path.abspath(load_model)
#         model_helper.load(load_model)
#
#     for i in range(7,14):
#         vloader = VoiceLoader(x_set, y_set,
#                               batch_size=16,
#                               n_mels=128,
#                               feature_pad_len=w,
#                               max_label_len=256,)
#                               # cut_sub=int(16*(1.5**i)), )#一步一步的扩大数据集，更容易拟合貌似
#
#         model_helper.fit(vloader,epoch=int(6*(1.5**i)))

# def train_relas(path,load_model = None):
#     '''
#     停止维护2019年7月1日，效果貌似很差，主要时间有限，以后有机会可以尝试一下
#         该方法因为类的变更，不保证能够运行成功
#     '''
#     w,h = 1024,128
#
#     thu_data = Thchs30(path)
#     x_set,y_set = thu_data.load_from_path()
#
#     model_helper = ReLASModel()
#     model_helper.compile(feature_shape=(w,h),ms_output_size=1437)
#     if load_model is not None:
#         load_model = os.path.abspath(load_model)
#         model_helper.load(load_model)
#
#     for i in range(14):
#         vloader = VoiceLoader(x_set, y_set,
#                               batch_size=16,
#                               n_mels=128,
#                               feature_pad_len=w,
#                               sil_mode=-1,
#                               max_label_len=256,
#                               cut_sub=int(16*(1.5**i)), )#一步一步的扩大数据集，更容易拟合貌似
#
#         model_helper.fit(vloader,epoch=int(6*(1.5**i)))

# def train_lasctc(path,load_model = None):
#     '''
#     停止维护2019年7月1日，跑不动
#         该方法因为类的变更，不保证能够运行成功
#     '''
#     w,h = 1024,128
#     max_label_len = 64
#
#     thu_data = Thchs30(path)
#     x_set,y_set = thu_data.load_from_path()
#
#     vloader = VoiceLoader(x_set,y_set,
#                           n_mels=128,feature_pad_len=w,
#                           max_label_len=max_label_len,
#                           cut_sub=16,
#                           sil_mode=-1,
#                           )
#
#     model_helper = LASCTCModel()
#     model_helper.compile(feature_shape=(w,h),label_max_string_length=max_label_len,ms_output_size=vloader.pymap.max_index)
#
#     if load_model is not None:
#         load_model = os.path.abspath(load_model)
#         model_helper.load(load_model)
#
#     model_helper.fit(vloader)
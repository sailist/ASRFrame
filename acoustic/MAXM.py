from core.base_model import AcousticModel
from keras.layers import Dense,Activation,Dropout,Input,Add
from core.ctc_function import CTC_Batch_Cost
from keras import Model
import os
from util.mapmap import PinyinMapper
from util.reader import VoiceDatasetList,VoiceLoader
from feature.mel_feature import MelFeature5

class MCONM(AcousticModel):
    '''将每一层的卷积连接起来的一次尝试,Somiao输入法到声学模型的迁移尝试
            2019年7月14日14:36:13，thchs30数据集上epoch=55，loss=59,基本无法下降，废弃
    '''
    def compile(self,feature_shape = (1024,200),label_max_string_length = 32,ms_output_size = 1423):
        audio_ipt = Input(name="audio_input", shape=feature_shape)


        parent_out = self.parent(audio_ipt,128)
        layer_h1 = self.conv1d_layers(audio_ipt,64,8)
        layer_h2 = self.cnn1d_cell(64, layer_h1, pool=False)
        layer_h3 = Add()([parent_out,layer_h2])

        # 64print(layer_h5)
        layer_h6 = Dropout(0.2)(layer_h3) # KL，双Dense
        layer_h7 = Dense(256, activation="relu", kernel_initializer="he_normal")(layer_h6) # TODO 考虑在这里加Attention
        layer_h7 = Dropout(0.2)(layer_h7)
        layer_h8 = Dense(ms_output_size)(layer_h7)
        y_pred = Activation(activation="softmax")(layer_h8)

        y_true = Input(name='label_inputs', shape=[label_max_string_length], dtype='float32')
        audio_length = Input(name='audio_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss_out = CTC_Batch_Cost()([y_true, y_pred, audio_length, label_length])
        train_model = Model([audio_ipt, y_true, audio_length, label_length], [loss_out])
        train_model.compile(optimizer="adam", loss={"ctc": lambda y_true, y_pred: y_pred})

        base_model = Model(audio_ipt, y_pred)

        self.built(train_model,base_model)

    @staticmethod
    def train(datagenes:list, load_model = None):

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


class MPCONM(AcousticModel):
    '''在MCONM的基础上将parent结构改为三层卷积+maxpool的尝试，其余条件相同
            2019年7月15日00:30:43，thchs30数据集上epoch=82，loss=14，此时下降已经变得有些困难，等待其继续训练，epoch>150次如果还未拟合则放弃
    '''
    def compile(self,feature_shape = (1024,200),label_max_string_length = 32,ms_output_size = 1423):
        audio_ipt = Input(name="audio_input", shape=feature_shape)

        parent_out = self.cnn1d_cell(32,audio_ipt,pool=True)
        parent_out = self.cnn1d_cell(64,parent_out,pool=True)
        parent_out = self.cnn1d_cell(64,parent_out,pool=True)

        layer_h1 = self.conv1d_layers(parent_out,64,8)
        layer_h2 = self.cnn1d_cell(64, layer_h1, pool=False)
        layer_h3 = Add()([parent_out,layer_h2])

        # 64print(layer_h5)
        layer_h6 = Dropout(0.2)(layer_h3) # KL，双Dense
        layer_h7 = Dense(256, activation="relu", kernel_initializer="he_normal")(layer_h6) # TODO 考虑在这里加Attention
        layer_h7 = Dropout(0.2)(layer_h7)
        layer_h8 = Dense(ms_output_size)(layer_h7)
        y_pred = Activation(activation="softmax")(layer_h8)

        y_true = Input(name='label_inputs', shape=[label_max_string_length], dtype='float32')
        audio_length = Input(name='audio_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss_out = CTC_Batch_Cost()([y_true, y_pred, audio_length, label_length])
        train_model = Model([audio_ipt, y_true, audio_length, label_length], [loss_out])
        train_model.compile(optimizer="adam", loss={"ctc": lambda y_true, y_pred: y_pred})

        base_model = Model(audio_ipt, y_pred)

        self.built(train_model,base_model)

    @staticmethod
    def train(datagenes: list, load_model=None):
        w, h = 1600, 200
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
                              divide_feature_len=8,
                              melf=MelFeature5(),
                              all_train=False
                              )

        model_helper = MPCONM(pymap)
        model_helper.compile(feature_shape=(w, h), label_max_string_length=max_label_len,
                             ms_output_size=pymap.max_index + 1)

        if load_model is not None:
            load_model = os.path.abspath(load_model)
            model_helper.load(load_model)

        model_helper.fit(vloader, epoch=-1, save_step=100, use_ctc=True)


class MPBCONM(AcousticModel):
    '''在MPCONM的基础上添加BatchNorm'''
    def compile(self,feature_shape = (1024,200),label_max_string_length = 32,ms_output_size = 1423):
        audio_ipt = Input(name="audio_input", shape=feature_shape)

        parent_out = self.cnn1d_cell(32,audio_ipt,pool=True)
        parent_out = self.cnn1d_cell(64,parent_out,pool=True)
        parent_out = self.cnn1d_cell(64,parent_out,pool=True)

        layer_h1 = self.conv1d_layers(parent_out,64,8,batch_norm=True)
        layer_h2 = self.cnn1d_cell(64, layer_h1, pool=False)
        layer_h3 = Add()([parent_out,layer_h2])

        # 64print(layer_h5)
        layer_h6 = Dropout(0.2)(layer_h3) # KL，双Dense
        layer_h7 = Dense(256, activation="relu", kernel_initializer="he_normal")(layer_h6) # TODO 考虑在这里加Attention
        layer_h7 = Dropout(0.2)(layer_h7)
        layer_h8 = Dense(ms_output_size)(layer_h7)
        y_pred = Activation(activation="softmax")(layer_h8)

        y_true = Input(name='label_inputs', shape=[label_max_string_length], dtype='float32')
        audio_length = Input(name='audio_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss_out = CTC_Batch_Cost()([y_true, y_pred, audio_length, label_length])
        train_model = Model([audio_ipt, y_true, audio_length, label_length], [loss_out])
        train_model.compile(optimizer="adam", loss={"ctc": lambda y_true, y_pred: y_pred})

        base_model = Model(audio_ipt, y_pred)

        self.built(train_model,base_model)

    @staticmethod
    def train(datagenes: list, load_model=None):
        w, h = 1600, 200
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
                              divide_feature_len=8,
                              melf=MelFeature5(),
                              all_train=False
                              )

        model_helper = MPBCONM(pymap)
        model_helper.compile(feature_shape=(w, h), label_max_string_length=max_label_len,
                             ms_output_size=pymap.max_index + 1)

        if load_model is not None:
            load_model = os.path.abspath(load_model)
            model_helper.load(load_model)

        model_helper.fit(vloader, epoch=-1, save_step=1000, use_ctc=True)

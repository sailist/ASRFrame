'''
本文件内所有模型涉及LSTM，训练非常慢，目前只在小数据集上拟合过，无法保证其在大数据集上的效果
'''
from keras.models import Model
import os
from util.reader import VoiceLoader,VoiceDatasetList
from util.mapmap import PinyinMapper
from feature.mel_feature import MelFeature5
from keras.layers import Conv1D,Dense, Dropout, Input, Reshape
from keras.layers import Bidirectional,LSTM,Activation
from core import AcousticModel,CTC_Batch_Cost

class LASModel(AcousticModel):
    def compile(self,feature_shape = (256,128),ms_output_size = 1423):
        self.ms_output_size = ms_output_size

        ipt = Input(shape=feature_shape, name="audio_input")
        layer_h1 = Bidirectional(LSTM(32, return_sequences=True), merge_mode="concat")(ipt)
        layer_h2 = Reshape((int(feature_shape[0]/2), feature_shape[1]))(layer_h1)
        layer_h3 = Bidirectional(LSTM(32, return_sequences=True), merge_mode="concat")(layer_h2)
        layer_h4 = Reshape((int(feature_shape[0]/4), feature_shape[1]))(layer_h3)

        layer_h4 = Conv1D(32, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(layer_h4)  # 卷积层
        layer_h4 = Dropout(rate=0.1)(layer_h4)
        layer_h5 = Conv1D(32, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(layer_h4)  # 卷积层

        layer_h6 = Dropout(rate=0.1)(layer_h5)
        layer_h7 = Conv1D(64, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(layer_h6)  # 卷积层

        layer_h7 = Dropout(rate=0.15)(layer_h7)
        layer_h8 = Conv1D(64, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(layer_h7)  # 卷积层

        layer_h9 = Dropout(0.15)(layer_h8)
        layer_h10 = Conv1D(128, 3,
                           activation='relu',
                           padding='same',
                           kernel_initializer='he_normal')(layer_h9)  # 卷积层

        layer_h10 = Dropout(0.2)(layer_h10)
        layer_h11 = Conv1D(128, 3,
                           activation='relu',
                           padding='same',
                           kernel_initializer='he_normal')(layer_h10)  # 卷积层

        layer_h12 = Dropout(0.2)(layer_h11)
        layer_h13 = Conv1D(128, 3,
                           activation='relu',
                           padding='same',
                           kernel_initializer='he_normal')(layer_h12)  # 卷积层
        layer_h13 = Dropout(0.3)(layer_h13)
        layer_h14 = Conv1D(128, 3,
                           activation='relu',
                           padding='same',
                           kernel_initializer='he_normal')(layer_h13)  # 卷积层

        print(layer_h14)
        layer_h17 = Dense(512, activation="relu",
                          kernel_initializer='he_normal')(layer_h14)  # 全连接层
        layer_h17 = Dropout(0.3)(layer_h17)
        layer_h18 = Dense(ms_output_size,
                          kernel_initializer='he_normal')(layer_h17)  # 全连接层

        y_pred = Activation('softmax', name='Activation0')(layer_h18)

        train_model = Model(ipt,y_pred)
        train_model.compile(optimizer="adam",loss="categorical_crossentropy")

        base_model = train_model

        self.built(train_model,base_model)

    # @staticmethod
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


class ReLASModel(AcousticModel):
    def compile(self,feature_shape = (256,128),ms_output_size = 1423):
        self.ms_output_size = ms_output_size

        ipt = Input(shape=feature_shape, name="audio_input")

        layer_h4 = Conv1D(32, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(ipt)  # 卷积层
        layer_h4 = Dropout(rate=0.1)(layer_h4)
        layer_h5 = Conv1D(32, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(layer_h4)  # 卷积层

        layer_h6 = Dropout(rate=0.1)(layer_h5)
        layer_h7 = Conv1D(64, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(layer_h6)  # 卷积层

        layer_h7 = Dropout(rate=0.15)(layer_h7)
        layer_h8 = Conv1D(64, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(layer_h7)  # 卷积层

        layer_h9 = Dropout(0.15)(layer_h8)
        layer_h10 = Conv1D(128, 3,
                           activation='relu',
                           padding='same',
                           kernel_initializer='he_normal')(layer_h9)  # 卷积层

        layer_h10 = Dropout(0.2)(layer_h10)
        layer_h11 = Conv1D(128, 3,
                           activation='relu',
                           padding='same',
                           kernel_initializer='he_normal')(layer_h10)  # 卷积层

        layer_h12 = Dropout(0.2)(layer_h11)
        layer_h13 = Conv1D(128, 3,
                           activation='relu',
                           padding='same',
                           kernel_initializer='he_normal')(layer_h12)  # 卷积层
        layer_h13 = Dropout(0.3)(layer_h13)
        layer_h14 = Conv1D(128, 3,
                           activation='relu',
                           padding='same',
                           kernel_initializer='he_normal')(layer_h13)  # 卷积层

        layer_h14 = self.pbilstm(layer_h14,32)
        layer_h14 = self.pbilstm(layer_h14,32)

        layer_h18 = Dense(ms_output_size,
                          kernel_initializer='he_normal')(layer_h14)  # 全连接层

        y_pred = Activation('softmax', name='Activation0')(layer_h18)

        train_model = Model(ipt,y_pred)
        train_model.compile(optimizer="adam",loss="categorical_crossentropy")

        base_model = train_model

        self.built(train_model,base_model)

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


class LASCTCModel(AcousticModel):
    def compile(self,feature_shape = (256,128),label_max_string_length = 32,ms_output_size = 1423):
        self.ms_output_size = ms_output_size

        ipt = Input(shape=feature_shape, name="audio_input")
        layer_h1 = Bidirectional(LSTM(32, return_sequences=True), merge_mode="concat")(ipt)
        layer_h2 = Reshape((int(feature_shape[0]/2), feature_shape[1]))(layer_h1)
        layer_h3 = Bidirectional(LSTM(32, return_sequences=True), merge_mode="concat")(layer_h2)
        layer_h4 = Reshape((int(feature_shape[0]/4), feature_shape[1]))(layer_h3)

        layer_h4 = Conv1D(32, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(layer_h4)  # 卷积层
        layer_h4 = Dropout(rate=0.1)(layer_h4)
        layer_h5 = Conv1D(32, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(layer_h4)  # 卷积层

        layer_h6 = Dropout(rate=0.1)(layer_h5)
        layer_h7 = Conv1D(64, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(layer_h6)  # 卷积层

        layer_h7 = Dropout(rate=0.15)(layer_h7)
        layer_h8 = Conv1D(64, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(layer_h7)  # 卷积层
        # layer_h8 = MaxPooling1D()(layer_h8)  # 池化层
        layer_h9 = Dropout(0.15)(layer_h8)
        layer_h10 = Conv1D(128, 3,
                           activation='relu',
                           padding='same',
                           kernel_initializer='he_normal')(layer_h9)  # 卷积层

        layer_h10 = Dropout(0.2)(layer_h10)
        layer_h11 = Conv1D(128, 3,
                           activation='relu',
                           padding='same',
                           kernel_initializer='he_normal')(layer_h10)  # 卷积层
        # layer_h11 = MaxPooling1D(pool_size=1, strides=None, padding="valid")(layer_h11)  # 池化层

        layer_h12 = Dropout(0.2)(layer_h11)
        layer_h13 = Conv1D(128, 3,
                           activation='relu',
                           padding='same',
                           kernel_initializer='he_normal')(layer_h12)  # 卷积层
        layer_h13 = Dropout(0.3)(layer_h13)
        layer_h14 = Conv1D(128, 3,
                           activation='relu',
                           padding='same',
                           kernel_initializer='he_normal')(layer_h13)  # 卷积层

        print(layer_h14)
        layer_h17 = Dense(512, activation="relu",
                          kernel_initializer='he_normal')(layer_h14)  # 全连接层
        layer_h17 = Dropout(0.3)(layer_h17)
        layer_h18 = Dense(ms_output_size,
                          kernel_initializer='he_normal')(layer_h17)  # 全连接层

        y_true = Input(name='label_inputs', shape=[label_max_string_length], dtype='float32')
        audio_length = Input(name='audio_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        print(layer_h18)

        y_pred = Activation('softmax', name='Activation0')(layer_h18)

        loss_out = CTC_Batch_Cost()([y_true, y_pred, audio_length, label_length])

        train_model = Model([ipt, y_true, audio_length, label_length], [loss_out])
        train_model.compile(optimizer="adam", loss={"ctc": lambda y_true, y_pred: y_pred})
        train_model.summary()

        base_model = Model(ipt,y_pred)

        self.built(train_model,base_model)

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
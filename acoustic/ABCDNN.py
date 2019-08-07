from keras.models import Model
from util.reader import VoiceDatasetList,VoiceLoader
from util.mapmap import PinyinMapper
from feature.mel_feature import *
import os
from keras.layers import Dense, Dropout, Input, Multiply, Conv2D, MaxPooling2D
from keras.layers import Activation
from core import AcousticModel, CTC_Batch_Cost
import keras.backend as K
class DCNN2D(AcousticModel):
    '''普通的2d卷积+maxpool，有一定的效果，但一般'''
    def compile(self,feature_shape = (256,128,1),label_max_string_length = 32,ms_output_size = 1242):
        '''
        建立模型[batch,times,vector,1]
        :param feature_shape:  音频特征形状[timestamp,feature_vec_dim]
        :param label_max_string_length: 标签最大长度
        :param ms_output_size: 输出范围
        :return:
        '''
        ipt = Input(name='audio_input', shape=feature_shape)

        # eb1 = self.encoder_block(audio_ipt,128,128,position_embedding=False)

        # eb2 = self.encoder_block(eb1,128,128,position_embedding = False)

        layer_h1 = Conv2D(32, 3,
                          use_bias=False,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(ipt)

        layer_h1 = Dropout(rate=0.05)(layer_h1)
        layer_h2 = Conv2D(32, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(layer_h1)

        layer_h2 = MaxPooling2D()(layer_h2)  # 池化层

        layer_h3 = Dropout(rate=0.05)(layer_h2)
        layer_h4 = Conv2D(64, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(layer_h3)  # 卷积层

        layer_h4 = Dropout(rate=0.1)(layer_h4)
        layer_h5 = Conv2D(64, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(layer_h4)  # 卷积层
        layer_h5 = MaxPooling2D()(layer_h5)  # 池化层

        layer_h6 = Dropout(rate=0.1)(layer_h5)
        layer_h7 = Conv2D(128, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(layer_h6)  # 卷积层

        layer_h7 = Dropout(rate=0.15)(layer_h7)
        layer_h8 = Conv2D(128, 3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(layer_h7)  # 卷积层
        layer_h8 = MaxPooling2D()(layer_h8)  # 池化层

        layer_h9 = Dropout(0.15)(layer_h8)
        layer_h10 = Conv2D(128, 3,
                           activation='relu',
                           padding='same',
                           kernel_initializer='he_normal')(layer_h9)  # 卷积层

        layer_h10 = Dropout(0.2)(layer_h10)
        layer_h11 = Conv2D(128, 3,
                           activation='relu',
                           padding='same',
                           kernel_initializer='he_normal')(layer_h10)  # 卷积层
        layer_h11 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h11)  # 池化层

        layer_h12 = Dropout(0.2)(layer_h11)
        layer_h13 = Conv2D(128,3,
                           activation='relu',
                           padding='same',
                           kernel_initializer='he_normal')(layer_h12)  # 卷积层
        layer_h13 = Dropout(0.3)(layer_h13)
        layer_h14 = Conv2D(128,3,
                           activation='relu',
                           padding='same',
                           kernel_initializer='he_normal')(layer_h13)  # 卷积层
        layer_h14 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h14)  # 池化层

        layer_h16 = self.reshape1dfrom2d(layer_h14)

        layer_h16 = Dropout(0.3)(layer_h16)
        layer_h17 = Dense(128, activation="relu",
                          kernel_initializer='he_normal')(layer_h16)  # 全连接层
        layer_h17 = Dropout(0.3)(layer_h17)
        layer_h18 = Dense(ms_output_size,
                          kernel_initializer='he_normal')(layer_h17)  # 全连接层

        y_pred = Activation('softmax', name='Activation0')(layer_h18)

        model_data = Model(inputs=ipt, outputs=y_pred)
        # model_data.summary()

        label_ipt = Input(name='label_inputs', shape=[label_max_string_length], dtype='float32')
        audio_length = Input(name='audio_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = CTC_Batch_Cost()([label_ipt,y_pred, audio_length, label_length])

        train_model = Model([ipt, label_ipt, audio_length, label_length], [loss_out])
        train_model.compile(optimizer="adam", loss={"ctc": lambda y_true, y_pred: y_pred})

        base_model = Model(ipt,y_pred)

        self.built(train_model,base_model)

    @staticmethod
    def train(datagenes: list, load_model=None):
        w, h = 1600, 200

        dataset = VoiceDatasetList()
        x_set, y_set = dataset.merge_load(datagenes)

        pymap = PinyinMapper(sil_mode=-1)
        vloader = VoiceLoader(x_set, y_set,
                              batch_size=16,
                              n_mels=h, feature_pad_len=w, feature_dim=3,
                              pymap=pymap,
                              melf=MelFeature5(),
                              divide_feature_len=8, )

        model_helper = DCNN2D(pymap)
        model_helper.compile(feature_shape=(w, h, 1),
                             ms_output_size=pymap.max_index + 1)  # ctcloss 计算要求： index < num_class-1

        if load_model is not None:
            load_model = os.path.abspath(load_model)
            model_helper.load(load_model)

        model_helper.fit(vloader, epoch=-1, use_ctc=True)

class DCBNN2D(AcousticModel):
    '''from https://github.com/audier/DeepSpeechRecognition
    2d卷积+maxpool+batchnorm，但效果不如1d的好，参数量也比1d的大
    '''
    def compile(self,feature_shape = (None,200,1),label_max_string_length = 32,ms_output_size = 1423):
        audio_ipt = Input(name="audio_input",shape=feature_shape)
        layer_h1 = self.cnn2d_cell(32, audio_ipt)
        layer_h2 = self.cnn2d_cell(64, layer_h1)
        layer_h3 = self.cnn2d_cell(128, layer_h2)
        layer_h4 = self.cnn2d_cell(128, layer_h3, pool=False)
        layer_h5 = self.cnn2d_cell(128, layer_h4, pool=False)

        layer_h6 = self.reshape1dfrom2d(layer_h5)

        layer_h6 = Dropout(0.2)(layer_h6)
        layer_h7 = Dense(256,activation="relu",kernel_initializer="he_normal")(layer_h6)
        layer_h7 = Dropout(0.2)(layer_h7)
        y_pred = Dense(ms_output_size, activation='softmax')(layer_h7)

        y_true = Input(name='label_inputs', shape=[label_max_string_length], dtype='float32')
        audio_length = Input(name='audio_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss_out = CTC_Batch_Cost()([y_true, y_pred, audio_length, label_length])
        train_model = Model([audio_ipt, y_true, audio_length, label_length], [loss_out])
        train_model.compile(optimizer="adam", loss={"ctc": lambda y_true, y_pred: y_pred})

        base_model = Model(audio_ipt,y_pred)

        self.built(train_model,base_model)

    @staticmethod
    def train(datagenes:list, load_model = None):
        w,h = 1600,200

        dataset = VoiceDatasetList()
        x_set, y_set = dataset.merge_load(datagenes)

        pymap = PinyinMapper(sil_mode=-1)
        vloader = VoiceLoader(x_set,y_set,batch_size=16,n_mels=h,feature_pad_len=w,feature_dim=3,cut_sub=32)

        model_helper = DCBNN2D(pymap)
        model_helper.compile(feature_shape=(w,h,1),label_max_string_length=32,ms_output_size=1423)

        if load_model is not None:
            load_model = os.path.abspath(load_model)
            model_helper.load(load_model)

        model_helper.fit(vloader)

class DCBNN1D(AcousticModel):
    '''当前（2019年7月1日）效果最好的一个模型,1d卷积+maxpool+batchnorm'''
    def compile(self,feature_shape = (1024,200),label_max_string_length = 32,ms_output_size = 1423):
        audio_ipt = Input(name="audio_input", shape=feature_shape)
        layer_h1 = self.cnn1d_cell(32, audio_ipt,pool=True,reshape=False)
        layer_h2 = self.cnn1d_cell(32, layer_h1,pool=True,reshape=False)
        layer_h3 = self.cnn1d_cell(64, layer_h2,pool=True,reshape=False)
        layer_h4 = self.cnn1d_cell(64, layer_h3, pool=False)
        layer_h5 = self.cnn1d_cell(128, layer_h4, pool=False)
        layer_h5 = self.cnn1d_cell(128, layer_h5, pool=False)
        layer_h5 = self.cnn1d_cell(128, layer_h5, pool=False)
        layer_h5 = self.cnn1d_cell(128, layer_h5, pool=False) # TODO 考虑多叠加几层

        # 64print(layer_h5)
        layer_h6 = Dropout(0.2)(layer_h5) # KL，双Dense
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
    def train(datagenes: list, load_model=None,**kwargs):
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
                              melf=MelFeature5(),
                              divide_feature_len=8,
                              all_train=False,
                              )

        model_helper = DCBNN1D(pymap)
        model_helper.compile(feature_shape=(w, h), label_max_string_length=max_label_len,
                             ms_output_size=pymap.max_index + 1)

        if load_model is not None:
            load_model = os.path.abspath(load_model)
            model_helper.load(load_model)


        epoch = kwargs.get("epoch",-1)
        save_step = kwargs.get("save_step",1000)


        model_helper.fit(vloader, epoch=epoch, save_step=save_step, use_ctc=True)

class DCBNN1Dplus(AcousticModel):

    def compile(self,feature_shape = (1024,200),label_max_string_length = 32,ms_output_size = 1423):
        audio_ipt = Input(name="audio_input", shape=feature_shape)
        layer_h1 = self.cnn1d_cell(32, audio_ipt,pool=True,reshape=False)
        layer_h2 = self.cnn1d_cell(32, layer_h1,pool=True,reshape=False)
        layer_h3 = self.cnn1d_cell(64, layer_h2,pool=True,reshape=False)
        layer_h4 = self.cnn1d_cell(64, layer_h3, pool=False)
        layer_h5 = self.cnn1d_cell(128, layer_h4, pool=False)
        layer_h5 = self.cnn1d_cell(128, layer_h5, pool=False)
        layer_h5 = self.cnn1d_cell(128, layer_h5, pool=False)
        layer_h5 = self.cnn1d_cell(128, layer_h5, pool=False) # TODO 考虑多叠加几层
        layer_h5 = self.cnn1d_cell(128, layer_h5, pool=False)
        layer_h5 = self.cnn1d_cell(128, layer_h5, pool=False)
        layer_h5 = self.cnn1d_cell(128, layer_h5, pool=False)
        layer_h5 = self.cnn1d_cell(128, layer_h5, pool=False)

        # 64print(layer_h5)
        layer_h6 = Dropout(0.2)(layer_h5) # KL，双Dense
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

class DCBANN1D(AcousticModel):
    def compile(self,feature_shape = (1024,200),label_max_string_length = 32,ms_output_size = 1423):
        audio_ipt = Input(name="audio_input", shape=feature_shape)
        layer_h1 = self.cnn1d_cell(32, audio_ipt,pool=True,reshape=False)
        layer_h2 = self.cnn1d_cell(32, layer_h1,pool=True,reshape=False)
        layer_h3 = self.cnn1d_cell(64, layer_h2,pool=True,reshape=False)
        layer_h4 = self.cnn1d_cell(64, layer_h3, pool=False)
        layer_h5 = self.cnn1d_cell(128, layer_h4, pool=False)
        layer_h5 = self.cnn1d_cell(128, layer_h5, pool=False)
        layer_h5 = self.cnn1d_cell(128, layer_h5, pool=False)
        layer_h5 = self.cnn1d_cell(128, layer_h5, pool=False) # TODO 考虑多叠加几层

        # 64print(layer_h5)
        layer_h6 = Dropout(0.2)(layer_h5) # KL，双Dense
        layer_h7 = Dense(512, activation="relu", kernel_initializer="he_normal")(layer_h6) # TODO 考虑在这里加Attention
        layer_h7 = Dropout(0.2)(layer_h7)

        attention_prob = Dense(units=512, activation='softmax', name='attention_vec')(layer_h7)
        attention_mul = Multiply()([layer_h7, attention_prob])

        layer_h8 = Dense(ms_output_size)(attention_mul)


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
                              melf=MelFeature5(),
                              divide_feature_len=8,

                              )

        model_helper = DCBANN1D(pymap)
        model_helper.compile(feature_shape=(w, h), label_max_string_length=max_label_len,
                             ms_output_size=pymap.max_index + 1)

        if load_model is not None:
            load_model = os.path.abspath(load_model)
            model_helper.load(load_model)

        model_helper.fit(vloader, epoch=-1, save_step=1000, use_ctc=True)

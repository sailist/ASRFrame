'''
本文件内所有模型涉及LSTM，训练非常慢，只在小数据集上拟合过，无法保证其在大数据集上的效果
'''
from keras.models import Model
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
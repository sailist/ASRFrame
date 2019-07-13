from core.base_model import AcousticModel
from keras.layers import Dense,Activation,Dropout,Input,Add
from core.ctc_function import CTC_Batch_Cost
from keras import Model

class MCONM(AcousticModel):
    '''将每一层的卷积连接起来的一次尝试,Somiao输入法到声学模型的迁移尝试'''
    def compile(self,feature_shape = (1024,200),label_max_string_length = 32,ms_output_size = 1423):
        audio_ipt = Input(name="audio_input", shape=feature_shape)
        parent_out = self.parent(audio_ipt,128)
        layer_h1 = self.conv1d_layers(audio_ipt,64,16)
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
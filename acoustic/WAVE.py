# wavenet 用于语音识别的keras实现，参考https://github.com/Deeperjia/tensorflow-wavenet/
from core.base_model import AcousticModel
from core.ctc_function import CTC_Batch_Cost
from keras.layers import Input,Conv1D,BatchNormalization,SeparableConv1D,Activation,Multiply,Add
from keras import Model

class WAVEM(AcousticModel):
    def compile(self,feature_shape=(None,200),label_max_string_length = 64,ms_output_size = 1438,h_dim = 128):


        ipt = Input(shape=feature_shape,name="mfcc_input")

        out = self.wave_conv1d_block(ipt,filters=h_dim)
        skip = None
        for _ in range(3):
            for rate in [1,2,4,8,16]:
                out,s = self.wave_residual_block(out,kernal_size=7,rate=rate,filters=h_dim)
                if skip is None:
                    skip = s
                else:
                    skip = Add()([skip,s])

        logit = self.wave_conv1d_block(skip,filters=64)
        y_pred = self.wave_conv1d_block(logit,filters=ms_output_size,bias=True,activation="softmax",)

        label_ipt = Input(name='label_inputs', shape=[label_max_string_length], dtype='float32')
        audio_length = Input(name='audio_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss_out = CTC_Batch_Cost()([label_ipt, y_pred, audio_length, label_length])

        train_model = Model([ipt, label_ipt, audio_length, label_length], [loss_out])
        train_model.compile(optimizer="adam", loss={"ctc": lambda y_true, y_pred: y_pred})

        base_model = Model(ipt, y_pred)

        self.built(train_model, base_model)



    def wave_residual_block(self,ipt,filters,kernal_size,rate):
        conv_out = self.dialted_conv(ipt,filters=filters,kernal_size=kernal_size,rate=rate,activation="tanh")
        conv_gate = self.dialted_conv(ipt,filters=filters,kernal_size=kernal_size,rate=rate,activation="sigmoid")
        out = Multiply()([conv_out,conv_gate])
        out = self.wave_conv1d_block(out,kernal_size=1,filters=filters)
        residual_out = Add()([out,ipt])
        return residual_out,out

    def wave_conv1d_block(self, ipt, kernal_size=1, filters=128, bias=False, activation="tanh"):
        out = Conv1D(filters=filters, kernel_size=kernal_size, padding="same",use_bias=bias)(ipt)
        if not bias:
            out = BatchNormalization()(out)
        out = Activation(activation=activation)(out)
        return out

    def dialted_conv(self,ipt,filters,kernal_size=7,rate=2,bias = False,activation="tanh"):
        out = Conv1D(filters=filters,
                     kernel_size=kernal_size,
                     use_bias=bias,
                     padding="same",
                     dilation_rate=rate,)(ipt)
        if not bias:
            out = BatchNormalization()(out)
        out = Activation(activation=activation)(out)
        return out

if __name__ == "__main__":
    wavenet = WAVEM(None)
    wavenet.compile(feature_shape=(1600,200))
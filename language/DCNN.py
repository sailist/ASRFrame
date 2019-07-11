from keras.layers import Input,Dense,Embedding,Dropout,Lambda
from keras.losses import categorical_crossentropy
from keras import Model
from core import LanguageModel

class DCNN1D(LanguageModel):
    '''比较差的语言模型，直觉上因为特征步长过少，卷积层增多时感受野太大忽略了局部特征，导致了该模型的失败
            2019年7月1日停止维护，使用更改后的SOMM
    '''
    def compile(self,feature_shape = (64,),ms_input_size = 1437,ms_output_size = 5000):
        ipt = Input(shape=feature_shape) # 拼音 index

        emb = self.cnn1d_cell(32,ipt,False)
        emb = self.cnn1d_cell(32,emb,False)
        # emb = self.cnn1d_cell(64,emb,False)
        # emb = self.cnn1d_cell(64,emb,False)
        # emb = self.cnn1d_cell(128,emb,False)
        # emb = self.cnn1d_cell(128,emb,False)

        emb = Dropout(rate=0.1)(emb)
        emb = Dense(512)(emb)
        emb = Dropout(rate=0.1)(emb)
        output = Dense(ms_output_size)(emb)

        y_true = Input(shape=(feature_shape[0],ms_output_size)) # 汉字 index
        label_len = Input(shape=(1,)) # 标签padding前长度
        # loss = Lambda(self._mask_categorical_crossentropy,name="mask_cc")([y_true,output,label_len])

        train_model = Model([ipt],[output])
        # train_model.compile(optimizer="adam",loss={"mask_cc":lambda y_true,y_pred:y_pred},metrics=["accuracy"])
        train_model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

        base_model = Model(ipt,output)
        self.built(train_model,base_model)

    def _mask_categorical_crossentropy(self,args):
        '''y_true 需要 to_categorical'''
        y_true, y_pred, label_len = args
        return categorical_crossentropy(y_true,y_pred)

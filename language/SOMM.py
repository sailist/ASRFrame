from keras.layers import Input,Dense,GRU,Bidirectional,Concatenate,Conv1D,MaxPooling1D,BatchNormalization,Dropout,Embedding,Add,Multiply,Lambda
from keras import Model
from keras.initializers import Constant
from core.base_model import LanguageModel

class SOMMword(LanguageModel):
    '''词粒度级的SOMM
        [pin] -> 拼音
    '''
    def compile(self,feature_shape = (200,),ms_pinyin_size = 1436,ms_output_size=8009,embed_size = 300,):
        py_ipt = Input(shape=feature_shape)
        emb = Embedding(ms_pinyin_size,embed_size)(py_ipt) # 注意其中的mask_zero选项 [batch,t,embvec]

        parent_out = self.parent(emb,embed_size) # [batch,t,emb//2]

        emb = self.conv1d_layers(parent_out,embed_size//2,5)
        emb = MaxPooling1D(pool_size=2,strides=1,padding="same")(emb)
        emb = Conv1D(filters=embed_size//2,kernel_size=5,padding="same")(emb)
        emb = BatchNormalization()(emb)
        emb = Conv1D(filters=embed_size//2,kernel_size=5,padding="same")(emb)
        emb = BatchNormalization()(emb)
        emb = Add()([emb,parent_out]) # Residual

        for i in range(4):
            emb = self.hignway_netblock(emb,embed_size//2)

        # emb = Bidirectional(GRU(embed_size//2,return_sequences=True),merge_mode="concat")(emb) # 因为要并行训练提升速度所以暂时不用该结构

        output = Dense(ms_output_size,activation="softmax")(emb)
        print(output)
        model = Model(py_ipt,output)
        model.compile(optimizer="adam",loss="categorical_crossentropy")

        model.summary()
        self.built(model,model)

class SOMMalpha(LanguageModel):
    '''字母粒度级的SOMM'''
    ''' pinyin -> 拼__音__ '''
    def compile(self,feature_shape = (200,),ms_pinyin_size = 1436,ms_output_size=8009,embed_size = 300,):
        py_ipt = Input(shape=feature_shape)
        emb = Embedding(ms_pinyin_size,embed_size)(py_ipt) # 注意其中的mask_zero选项 [batch,t,embvec]

        parent_out = self.parent(emb,embed_size)

        emb = self.conv1d_layers(parent_out,embed_size//2,5)
        emb = MaxPooling1D(pool_size=2,strides=1,padding="same")(emb)
        emb = Conv1D(filters=embed_size//2,kernel_size=5,padding="same")(emb)
        emb = BatchNormalization()(emb)
        emb = Conv1D(filters=embed_size//2,kernel_size=5,padding="same")(emb)
        emb = BatchNormalization()(emb)
        emb = Add()([emb,parent_out]) # Residual

        for i in range(4):
            emb = self.hignway_netblock(emb,embed_size//2)

        # emb = Bidirectional(GRU(embed_size//2,return_sequences=True),merge_mode="concat")(emb) # 因为要并行训练提升速度所以暂时不用该结构

        output = Dense(ms_output_size,activation="softmax")(emb)
        print(output)
        model = Model(py_ipt,output)
        model.compile(optimizer="adam",loss="categorical_crossentropy")

        model.summary()
        self.built(model,model)


if __name__ == "__main__":
    s = SOMMalpha()
    s.compile(ms_pinyin_size=1436,ms_output_size=8009)
    pass
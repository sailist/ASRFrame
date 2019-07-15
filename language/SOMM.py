from keras.layers import Input,Dense,Conv1D,MaxPooling1D,BatchNormalization,Embedding,Add
from keras import Model
from core.base_model import LanguageModel
from util.reader import TextDataGenerator,TextLoader2,VoiceDatasetList,TextLoader

from util.mapmap import PinyinMapper,ChsMapper
from keras.preprocessing.sequence import pad_sequences

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

    @staticmethod
    def train(datagene: list, load_model=None):

        dataset = VoiceDatasetList()
        _, y_set = dataset.merge_load(datagene, choose_x=False, choose_y=True)

        max_label_len = 64

        pinyin_map = PinyinMapper(sil_mode=0)
        chs_map = ChsMapper()

        tloader = TextLoader(y_set, padding_length=max_label_len, pinyin_map=pinyin_map, chs_map=chs_map)

        model_helper = SOMMword()
        model_helper.compile(feature_shape=(max_label_len,),
                             ms_pinyin_size=pinyin_map.max_index,
                             ms_output_size=chs_map.categores)

        if load_model is not None:
            model_helper.load(load_model)

        model_helper.fit(tloader, -1)


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

    @staticmethod
    def train(datagene: TextDataGenerator, load_model=None):

        txtfs = datagene.load_from_path()

        max_label_len = 200

        pinyin_map = PinyinMapper(sil_mode=0)
        chs_map = ChsMapper()

        tloader = TextLoader2(txtfs, padding_length=max_label_len, pinyin_map=pinyin_map, chs_map=chs_map,
                              grain=TextLoader2.grain_alpha,
                              cut_sub=175,
                              )

        model_helper = SOMMalpha()
        model_helper.compile(feature_shape=(max_label_len,),
                             ms_pinyin_size=pinyin_map.max_index,
                             ms_output_size=chs_map.categores)

        if load_model is not None:
            model_helper.load(load_model)

        model_helper.fit(tloader, -1)


    @staticmethod
    def real_predict(path):
        max_label_len = 200
        pinyin_map = PinyinMapper(sil_mode=0)
        chs_map = ChsMapper()

        model_helper = SOMMalpha()
        model_helper.compile(feature_shape=(max_label_len,),
                             ms_pinyin_size=pinyin_map.max_index,
                             ms_output_size=chs_map.categores)

        model_helper.load(path)

        while True:
            string = input("请输入拼音:")
            xs = [pinyin_map.alist2vector(string)]
            print(xs)
            batch = pad_sequences(xs,maxlen=max_label_len,padding="post",truncating="post"),None
            result = model_helper.predict(batch)[0]
            print(result.replace("_",""))


if __name__ == "__main__":

    # 运行文件查看网络结构
    SOMMword().compile(ms_pinyin_size=1436,ms_output_size=8009)
    SOMMalpha().compile(ms_pinyin_size=1436,ms_output_size=8009)
    pass
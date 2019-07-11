import os
from scipy.io import wavfile
from keras.initializers import Constant
from keras.layers import Dense,Concatenate,Dropout,Add,Multiply,Lambda
from keras.layers import BatchNormalization,MaxPooling2D,Conv2D,Reshape,Conv1D,MaxPooling1D,Bidirectional,LSTM
import keras.backend as K
from keras import Model
from util.reader import VoiceLoader,TextLoader,TextLoader2
from util.evaluate import EvaluateDistance
from core.ctc_function import CTCDecode
import config
from core.position_embedding import Position_Embedding
from core.glu import GatedConv1D
from core.attention import Attention
from util.callbacks import Lossplot,TimeClock
from util import PinyinMapper
from util.mapmap import ChsMapper
from util.audiotool import Recoder, NoiseFilter, VadExtract
from feature.mel_feature import MelFeature5

class BaseModel():
    '''
    compile方法和自己写，其他方法通过实现固定接口可以使用父类的
    对compile方法的要求：
        必须建立两个模型，一个是train_model,一个是base_model
        fit时使用train_model训练，输入输出自定义
        predict时使用base_model，要求base_model必须以xs（音频特征序列）为输入，以y_pred（拼音的index序列）为输出，方便调用try_predict方法
    '''
    def __init__(self):
        self.model_built = False
        self.evaluate = EvaluateDistance()
        self.train_model = None
        self.base_model = None

    def built(self,train_mode:Model,base_model:Model):
        train_mode.summary()

        self.train_model = train_mode
        self.base_model = base_model

        self.model_built = True
        print('[Info] Create Model Successful, Compiles Model Successful.')

    def save(self,dir_path = None,fn=None):
        '''
        :param dir_path: the model storeed path, if path not existed, it will be created
            so you have no need to check whether the dir is existed.
        :param fn:str or int，used to store model，
        if is str，it will be the model name
        if is int,it will auto be formated to "{self.__class__.__name__}_step_{fn}.h5"
        :return:
        '''
        assert self.model_built, "model have not built, please excute compile() method to build the model."
        if dir_path is None:
            dir_path = config.model_dir

        os.makedirs(dir_path, exist_ok=True)
        if fn is None:
            fn = "trans_model.h5"

        if isinstance(fn,int):
            fn = f"{self.__class__.__name__}_step_{fn}.h5"

        save_path = os.path.join(dir_path, fn)
        self.train_model.save(save_path)
        print(f"saved model, model name:{fn}")

    def load(self, path):
        '''
        load the model weight, please be sure excuting compile() method before load().
        :param path: the path of the model file
        :return:
        '''
        assert self.model_built, "model have not built, please excute compile() method to build the model."
        self.train_model.load_weights(path)
        self.base_model.load_weights(path)

    def catch_sublayer(self,layer_name):
        '''TODO 未经过测试'''
        if isinstance(layer_name,int):
            return K.function([self.train_model.layers[0].input], [self.train_model.layers[1].output])
        elif isinstance(layer_name,str):
            return K.function([self.train_model.layers[0].input],[self.train_model.get_layer(layer_name).output])

    def cnn2d_cell(self, size, x, pool = True,batch_norm = True):
        '''
        2D卷积+批量归一化+池化组合
        :param size: 卷积核数目
        :param x: 输入，(batch，w,h,channle)
        :param pool:
        :param reshape:
        :return:
        '''
        if batch_norm:
            x = BatchNormalization()(Conv2D(size,3,padding="same",kernel_initializer="he_normal")(x))
            x = BatchNormalization()(Conv2D(size,3,padding="same",kernel_initializer="he_normal")(x))
        else:
            x = Conv2D(size, 3, padding="same", kernel_initializer="he_normal")(x)
            x = Conv2D(size, 3, padding="same", kernel_initializer="he_normal")(x)

        if pool:
            x = MaxPooling2D()(x)
        return x

    def cnn1d_cell(self,size,x,pool = True,reshape = False,batch_norm = True):
        '''
        1D卷积+批量归一化+池化/Reshape组合

        :param size: int, number of filters
        :param x: Tensor,(batch,timestamp,vector)
        :param pool: bool, whether to use maxpool
        :param reshape: bool, whether to use reshape, like [1,2,3,4] -> [[1,2],[3,4]]
        :return:
        '''
        if batch_norm:
            x = BatchNormalization()(Conv1D(size, 3, padding="same",activation="relu", kernel_initializer="he_normal")(x))
            x = BatchNormalization()(Conv1D(size, 3, padding="same",activation="relu", kernel_initializer="he_normal")(x))
        else:
            x = Conv1D(size, 3, padding="same",activation="relu", kernel_initializer="he_normal")(x)
            x = Conv1D(size, 3, padding="same",activation="relu", kernel_initializer="he_normal")(x)

        if pool:
            x = MaxPooling1D()(x)
        elif reshape:
            x_shape = x.shape.as_list() # [batch,timestamp,size]

            x = Reshape((int(x_shape[1]/2),size*2))(x)

        return x

    def gatecnn1d_cell(self,size,x,pool = True,batch_norm = True):
        '''参考 Gated conv'''
        if batch_norm:
            x = BatchNormalization()(GatedConv1D(size,3,padding="same")(x))
        else:
            x = GatedConv1D(size, 3, padding="same")(x)

        if pool:
            x = MaxPooling1D()(x)

        return x

    def pbilstm(self,x,filter):
        '''LAS中的listener结构，https://arxiv.org/abs/1508.01211'''
        x_shape = x.shape.as_list()

        x = Bidirectional(LSTM(filter, return_sequences=True), merge_mode="concat")(x)
        x = Reshape((int(x_shape[1]/2),filter*4))(x)

        # x = Reshape((int(x_shape[1] / 2), x_shape[2]*2))(x)
        return x

    def reshape1dfrom2d(self,x):
        '''手动将2D矩阵reshape到1d矩阵'''
        x_shape = x.shape.as_list()
        x = Reshape((x_shape[1],x_shape[2]*x_shape[3]))(x)
        return x

    def attention_block(self,next_ipt,attention_dim = 32,h_dim = 32,position_embedding = True):
        '''
        Transformer中的attention结构，positional embedding + Attention + Conv1D，但貌似效果不好，目前暂时不清楚原因
        :param next_ipt:输入，[batch,time_step,hdim]
        :param attention_dim: int value, used in Attention Layer
        :param h_dim: int value, conv Layer's filters' number
        :param position_embedding:
        :return:
        '''
        if position_embedding:
            next_ipt = Position_Embedding(2, mode="concat")(next_ipt)

        att1 = Attention(attention_dim)([next_ipt,next_ipt,next_ipt])
        # att1 = LayerNormalization()(att1)
        conv1 = Conv1D(h_dim,3,padding="same",kernel_initializer="he_normal")(att1)

        return conv1

class AcousticModel(BaseModel):
    '''继承自BaseModel，用于训练声学模型，主要区别在于save时候目录存放位置和fit、predict、test方法不同'''
    def __init__(self,pymap):
        super().__init__()
        self.pymap = pymap
        self.pysets = set()
        self.ctc_decoder = CTCDecode()

    def save(self, dir_path=None, fn=None, latest=3):
        if dir_path is None:
            dir_path = config.acoustic_model_dir
        super().save(dir_path, fn)

    def fit(self, voice_loader:VoiceLoader, epoch = 100, save_step = 500,use_ctc = False):
        '''
        传入数据生成器，执行训练
        :param voice_loader:参考VoiceLoader的使用方法
        :param epoch: 训练多少次epoch，由代码手动控制而不是设置在kears的训练中，如果为-1，则为一直训练
        :param save_step: 每一个epoch训练多少步，这里每一个epoch训练完成后就会保存一次模型
        :param use_ctc: 该模型测试时是否使用ctc解码对输出进行处理，注意是测试的时候，该参数不影响训练过程
        :return:
        '''
        # viter = voice_loader.create_feature_iter(shuffle_set=False)

        self.pymap = voice_loader.pymap

        i = -1
        self.save(fn=(i+1)*save_step)
        self.test(voice_loader.choice(), use_ctc=use_ctc)

        logg_plot = Lossplot(self.__class__.__name__, save_dir=config.acoustic_loss_dir)
        time_clock = TimeClock()
        while i<epoch or epoch == -1:
            i+=1
            print(f"train epoch {i}/{epoch}.")
            self.train_model.fit_generator(voice_loader,save_step,callbacks=[logg_plot,time_clock])

            self.test(voice_loader.choice_test(), use_ctc=use_ctc)
            self.save(fn=(i+1)*save_step)

    def predict(self, batch, use_ctc = True,return_ctc_prob = False):
        '''
        预测一个batch，直接返回拼音
        :param batch: 要求从Voiceloader中获取，或者满足格式[xs,None,feature_len,None],None
        :param use_ctc: 是否是用ctc解码
        :return: 拼音list，[batch,pylist]
        '''
        [xs, _, feature_len, _], _ = batch
        prob = None

        prob_result = self.prob_predict(batch)
        if use_ctc:
            argmax_res = self.ctc_decoder.ctc_decode(prob_result, feature_len,return_ctc_prob)
            if return_ctc_prob:
                argmax_res,prob = argmax_res
        else:
            argmax_res = K.argmax(prob_result)
            argmax_res = K.eval(argmax_res)

        pylist_pred = self.pymap.batch_vector2pylist(argmax_res, return_word_list=True, return_list=True)

        if return_ctc_prob:
            return pylist_pred,prob
        return pylist_pred

    def prob_predict(self, batch):
        '''
        预测一个batch，返回经过softmax后的概率
        :param batch: 要求从Voiceloader中获取，或者满足格式[xs,None,feature_len,None],None
        :param use_ctc:
        :return:
        '''
        [xs, _, _, _], _ = batch
        result = self.base_model.predict(xs)
        return result

    def test(self, batch, use_ctc = False,return_result = False):
        '''
        要求self.base_model 接受的输入必须是xs(根据模型不同维度可以任意),输出必须是ys_pred（未经过argmax的,[batch,timestamp,vector]）
        当然，你也可以自行实现自己的try_predict方法

        :param batch: 要求格式必须是 [xs, ys, feature_len, label_len], placehold

                其中，由于不同模型的要求不同，ys同时支持catagries维（batch,timestamp,num_catagries）和index向量(batch,indexs)，会根据维度自动进行判断
                result 的输出维度必须符合 [batch,timestamp,num_categries]
        :param return_result: 是在控制台输出还是返回结果，与model_summary.py对应
        :return:
        '''
        [xs, ys, feature_len , label_len], placehold = batch

        result = self.base_model.predict(xs)

        # assert use_ctc and result.ndim == 3,"when use_ctc is True, result.ndim must be 3. please check params."
        if use_ctc:
            argres = self.ctc_decoder.ctc_decode(result,feature_len)
        else:
            argres = K.argmax(result)
            argres = K.eval(argres)

        if ys.ndim == 3:
            y_true = K.argmax(ys)
            y_true = K.eval(y_true)
        else:
            y_true = ys

        pylist_pred = self.pymap.batch_vector2pylist(argres, return_word_list=True, return_list=True)
        pylist_true = self.pymap.batch_vector2pylist(y_true, return_word_list=True, return_list=True)
        print("===================")
        all_count = 0
        all_norm = 0
        ignore_num = 0
        i = 0

        err_dict = {}

        for pred, true, llen in zip(pylist_pred, pylist_true,label_len.squeeze()):
            true = true[:llen]
            count,count_norm = self.evaluate.compare_sent(pred, true)

            if count == 0:
                self.pysets.update(pred)
            if len(pred) == len(true):
                for a,b in zip(true,true):
                    if a != b:
                        errlist = err_dict.setdefault(a,[])
                        errlist.append(b)
                        print(a,b)
            else:
                print(" ".join(pred))
                print(" ".join(true))
                ignore_num+=1


            all_count += count
            all_norm += count_norm
            i+=1

            if not return_result:
                print(" ".join(pred))
                print(" ".join(true))
                print("-------------------")
                print(f"[test*] compare result:{count} differences. After norm:{count_norm}. ")
                print("===================")

        if not return_result:
            print(f"[test*] all differences:{all_count}.Whole norm:{all_norm/i}")
            print(f"[info*] pinyin can recognition:{len(self.pysets)}")

        print(err_dict,ignore_num)

        return {
            "all_count":all_count,
            "all_norm":all_norm/i,
            "err_pylist":err_dict,
            "ignore":ignore_num,
        }

class LanguageModel(BaseModel):
    '''继承自BaseModel，用于训练声学模型，主要区别在于save时候目录存放位置和fit、predict、test方法不同'''
    def __init__(self):
        super().__init__()
        self.chs_map = ChsMapper()

    def save(self, dir_path=None, fn=None):
        if dir_path is None:
            dir_path = config.language_model_dir
        super().save(dir_path, fn)

    def fit(self, txt_loader:[TextLoader,TextLoader2], epoch = 100, save_step = 500):
        '''
        传入数据生成器，执行训练
        :param txt_loader:参考VoiceLoader的使用方法
        :param epoch: 训练多少次epoch，由代码手动控制而不是设置在kears的训练中，如果为-1，则为一直训练
        :param save_step: 每一个epoch训练多少步，这里每一个epoch训练完成后就会保存一次模型
        :param use_ctc: 该模型测试时是否使用ctc解码对输出进行处理，注意是测试的时候，该参数不影响训练过程
        :return:
        '''
        # viter = voice_loader.create_feature_iter(shuffle_set=False)


        i = -1
        self.save(fn=(i+1)*save_step)
        self.test(txt_loader.choice())

        logg_plot = Lossplot(self.__class__.__name__,save_dir=config.language_loss_dir)
        time_clock = TimeClock()
        while i<epoch or epoch == -1:
            i+=1
            print(f"train epoch {i}/{epoch}.")
            self.train_model.fit_generator(txt_loader, save_step, callbacks=[logg_plot, time_clock])

            self.test(txt_loader.choice())
            self.save(fn=(i+1)*save_step)
            # self.save_loss_plot(None) # TODO 实现损失曲线的绘制，每个模型一个，覆盖原图片，默认保存在 ./loss_plot 下

    def test(self,batch):
        # [xs, ys, label_len], placeholder = batch
        xs, ys = batch
        result = self.base_model.predict(xs)

        result = K.argmax(result)
        result = K.eval(result)

        ys = K.argmax(ys)
        ys = K.eval(ys)

        result = self.chs_map.batch_vector2chsent(result)
        ys = self.chs_map.batch_vector2chsent(ys)
        # for pre_line,true_line in zip(result,ys):
        #     print("————————————————————————")
        #     print(pre_line)
        #     print(true_line)
        #     count, count_norm = self.evaluate.compare_sent(pre_line, true_line)

        print("===================")
        all_count = 0
        all_norm = 0
        i = 0
        for pred, true in zip(result, ys):
            count, count_norm = self.evaluate.compare_sent(pred, true)

            all_count += count
            all_norm += count_norm
            i += 1
            print("".join(pred).replace("_",""))
            print("".join(true).replace("_",""))
            print("-------------------")
            print(f"[test*] compare result:{count} differences. After norm:{count_norm}. ")
            print("===================")
        print(f"[test*] all differences:{all_count}.Whole norm:{all_norm/i}")

        # self.chs_map.vector2chsent()

    def predict(self,batch,return_prob = False):
        xs,_ = batch
        prob = self.base_model.predict(xs)

        result = K.argmax(prob)
        result = K.eval(result)
        if return_prob:
            return self.chs_map.batch_vector2chsent(result),prob
        return self.chs_map.batch_vector2chsent(result)


    def blur_predict(self,batch):
        xs,ys = batch
        prob = self.base_model.predict(xs)

        result = K.argmax(prob)

        result = K.eval(result)

        return self.chs_map.batch_vector2chsent(result),prob

    # def _judge_blur_range(self,batch,thresh = 0.01):
    #     for sample in batch:

    def hignway_netblock(self,x,h_dim):
        H = Dense(h_dim,activation="relu")(x)
        T = Dense(h_dim,activation="sigmoid",kernel_initializer=Constant(value=-1))(x)

        C = Lambda(lambda x:1-x)(T)

        A = Multiply()([H,T])
        B = Multiply()([x,C])
        outputs = Add()([A,B])

        return outputs

    def parent(self,ipt,h_dim,drop_out_rate = 0.5):
        emb = Dense(h_dim)(ipt)
        emb = Dropout(rate=drop_out_rate)(emb)
        emb = Dense(h_dim//2)(emb)
        emb = Dropout(rate=drop_out_rate)(emb)
        return emb

    def conv1d_layers(self,x,h_dim,layer_num = 16):
        embs = Conv1D(h_dim,kernel_size=1,padding="same")(x)#[emb/2]
        for i in range(2,layer_num+1):
            emb = Conv1D(h_dim,kernel_size=i,padding="same")(x) #[emb/2]
            embs = Concatenate()([embs,emb]) # axis = -1#[emb]+[emb//2]
        embs = BatchNormalization()(embs) #[emb]

        return embs

class BaseJoint():
    '''
    需要自己实现compile、raw_record等方法
    '''
    def __init__(self, acmodel_input_shape, acmodel_output_shape, lgmodel_input_shape, py_map:PinyinMapper, chs_map:ChsMapper,**kwargs):
        '''

        :param acmodel_input_shape: [timestamp,sample]
        :param acmodel_output_shape: [index,] or [timestamp,prob after softmax]
        :param lgmodel_input_shape: [index,]
        :param py_map:
        :param chs_map:
        '''
        self.vad_extract = VadExtract()
        self.melf = MelFeature5()
        self.audio_tool = NoiseFilter()
        self.acmodel_input_shape = acmodel_input_shape
        self.acmodel_output_shape = acmodel_output_shape
        self.lgmodel_input_shape = lgmodel_input_shape
        self.py_map = py_map
        self.chs_map = chs_map
        self.divide_feature = kwargs.get("divide_feature",1)


    def pre_process_audio(self,xs):
        '''
        通用的预处理方法，因为有可能不通用因此需要在voice_rec里手动调用，进行特征提取和padding
        其中padding的大小为acmodel_input_shape[0]
        :param xs: (sample,),就是录音完后的一串向量
        :return:
            xs:[batch,padding_len,sample]
            feature_len:[batch, raw_feature_len]
        '''

        xs = self.vad_extract.audio2batch_by_extract(xs)  # [batch,sub_xs]

        # xs:[batch,timestamp,sample] feature_len:[batch,sample]
        xs, feature_len = VoiceLoader.audio2feature(xs,
                                                    self.acmodel_input_shape[0],
                                                    self.divide_feature,
                                                    self.melf)
        return xs,feature_len

    def raw_record(self,xs):
        raise NotImplementedError(f"You need to implement your own raw_record method in class {self.__class__.__name__}")

    def record_from_wav(self,fs):
        sr,xs = wavfile.read(fs)
        return self.raw_record(xs)

    def record_from_cmd(self, second = 5, use_ctc = True):
        xs = Recoder.record(second)
        return self.raw_record(xs)
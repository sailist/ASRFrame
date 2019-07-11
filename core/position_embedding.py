from keras import backend as K
from keras.engine.topology import Layer

class Position_Embedding(Layer):
    '''
    Position Embedding
    使用:
        V = ...code...
        pe = Position_Embedding(embedding_size,mode="concat")(V)
    '''
    def __init__(self, size=None, mode='sum', **kwargs):
        '''

        :param size: 为偶数，选择每一个时间步的位置向量的长度，如果和不指定，则默认为和每一个时间步的向量的长度相同
        :param mode: 'sum' or 'concat' ,选择位置向量拼接的方式是求和还是拼接，在原论文中有描述，貌似没有区别
        :param kwargs: 其他参数，默认不需要
        '''
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x,**kwargs):
        '''
        :param x: [batch,w,h]
        :return: 如果mode == 'sum'，则输出维度仍然为[batch,w,h]，若mode == 'concat'，则输出维度为[batch,w,h+size]
        '''
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  # K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)



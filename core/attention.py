from keras import backend as K
from keras.engine.topology import Layer

class Attention(Layer):
    '''
    只有单个Attention层,且经过一层 线性变换 ，可以映射到任意维度

    使用：
        Q = ...code...
        K = ...code...
        V = ...code...

        att = Attention(h_dim)([Q,K,V])
    '''
    def __init__(self, h_dim, **kwargs):
        '''
        :param h_dim:  输出后每个时间步的向量长度
        :param kwargs:
        '''
        self.h_dim = h_dim
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.h_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.h_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.h_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)


    def call(self, x,**kwargs):
        '''
        :param x: Q,K,V，其中当Q=K=V时，即为self-attention
            Q:[batch_size,time_step,dq]
            K:[batch_size,time_step,dk]
            V:[batch_size,time_step,dv]
        :return: Attention后的C ,[batch_size,time_ste,h_dim]
        '''

        assert len(x) == 3, f"input dim must be 3,but {len(x)}" # 确保输入为三

        Q_seq, K_seq, V_seq = x


        # 对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)  #[batch_size,time_step,h_dim],因为W 非batch，因此用K.dot(),K会自动的将每个样本和W相乘
        K_seq = K.dot(K_seq, self.WK)
        V_seq = K.dot(V_seq, self.WV)

        # 计算Q,K的内积（即计算相似度，这一步可以替换为其他的计算函数）,然后softmax
        A = K.batch_dot(Q_seq, K.permute_dimensions(K_seq, [0, 2, 1])) / self.h_dim ** 0.5

        # 计算相似度后进行softmax归一化
        A = K.softmax(A)

        # 输出
        O_seq = K.batch_dot(A, V_seq)
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.h_dim)


class RestrictedAttention(Layer):
    '''
    TODO 约束Attention，每一个K相应的时间步，只看前后n个时间步的，其余的设置为负无穷，这样softmax后得到的为0，使得Attention更集中
    该操作同时减少了复杂度，在《Attention is all you need》中提到过，但是貌似没有被使用。
    '''
    pass

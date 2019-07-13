from keras import backend as K
from keras.layers import Lambda,Input
from keras import Model
from tensorflow.python.ops import ctc_ops as ctc
import tensorflow as tf


class CTC_Batch_Cost():
    '''
    用于计算CTC loss
    '''
    def ctc_lambda_func(self,args):
        """Runs CTC loss algorithm on each batch element.

        # Arguments
            y_true: tensor `(samples, max_string_length)` 真实标签
            y_pred: tensor `(samples, time_steps, num_categories)` 预测前未经过softmax的向量
            input_length: tensor `(samples, 1)` 每一个y_pred的长度
            label_length: tensor `(samples, 1)` 每一个y_true的长度

            # Returns
                Tensor with shape (samples,1) 包含了每一个样本的ctc loss
            """
        y_true, y_pred, input_length, label_length = args

        # y_pred = y_pred[:, :, :]
        # y_pred = y_pred[:, 2:, :]
        return self.ctc_batch_cost(y_true, y_pred, input_length, label_length)

    def __call__(self, args):
        '''
        ctc_decode 每次创建会生成一个节点，这里参考了https://blog.csdn.net/u014484783/article/details/88849971
        将ctc封装成模型，是否会解决这个问题还没有测试过这种方法是否还会出现创建节点的问题
        '''
        y_true = Input(shape=(None,))
        y_pred = Input(shape=(None,None))
        input_length = Input(shape=(1,))
        label_length = Input(shape=(1,))

        lamd = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([y_true,y_pred,input_length,label_length])
        model = Model([y_true,y_pred,input_length,label_length],[lamd],name="ctc")

        # return Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')(args)
        return model(args)

    def ctc_batch_cost(self,y_true, y_pred, input_length, label_length):
        """Runs CTC loss algorithm on each batch element.

        # Arguments
            y_true: tensor `(samples, max_string_length)`
                containing the truth labels.
            y_pred: tensor `(samples, time_steps, num_categories)`
                containing the prediction, or output of the softmax.
            input_length: tensor `(samples, 1)` containing the sequence length for
                each batch item in `y_pred`.
            label_length: tensor `(samples, 1)` containing the sequence length for
                each batch item in `y_true`.

        # Returns
            Tensor with shape (samples,1) containing the
                CTC loss of each element.
        """
        label_length = tf.to_int32(tf.squeeze(label_length, axis=-1))
        input_length = tf.to_int32(tf.squeeze(input_length, axis=-1))
        sparse_labels = tf.to_int32(K.ctc_label_dense_to_sparse(y_true, label_length))

        y_pred = tf.log(tf.transpose(y_pred, perm=[1, 0, 2]) + 1e-7)

        # 注意这里的True是为了忽略解码失败的情况，此时loss会变成nan直到下一个个batch
        return tf.expand_dims(ctc.ctc_loss(inputs=y_pred,
                                           labels=sparse_labels,
                                           sequence_length=input_length,
                                           ignore_longer_outputs_than_inputs=True), 1)

class CTCDecode():
    '''用与CTC 解码，得到真实语音序列'''
    def __init__(self):
        # base_pred = Input(shape=[None,None])
        # feature_len = Input(shape=[1,])
        # decode = Lambda(self._ctc_decode)([base_pred,feature_len])
        # self.model = Model([base_pred,feature_len],[decode])
        pass

    def _ctc_decode(self,args):
        base_pred, in_len = args
        in_len = K.squeeze(in_len,axis=-1)
        r = K.ctc_decode(base_pred, in_len, greedy=True, beam_width=100, top_paths=1)
        r1 = K.eval(r[0][0])
        prob = K.eval(r[1][0])
        # return r[:,0]
        return r1,prob

    def ctc_decode(self,base_pred,in_len,return_prob = False):
        '''
        :param base_pred:
        :param in_len: [sample,1]
        :return:
        '''
        # result, prob = self.model.predict([base_pred,in_len])
        result, prob = self._ctc_decode([base_pred,in_len])
        print(prob)
        if return_prob:
            return result,prob
        return result

    def __call__(self,base_pred,in_len,return_prob = False):
        return self.ctc_decode(base_pred,in_len,return_prob)


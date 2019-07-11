from keras.layers import Conv1D,Multiply


class GatedConv1D():
    '''门控线性单元 https://arxiv.org/abs/1612.08083'''
    def __init__(self, filters, kernel_size, strides=1, padding='same',kernel_initializer = "he_normal"):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer

    def call(self,x):
        A = Conv1D(self.filters,
                   kernel_size=self.kernel_size,
                   padding=self.padding,
                   strides=self.strides,
                   kernel_initializer=self.kernel_initializer)(x)
        B = Conv1D(self.filters,
                   kernel_size=self.kernel_size,
                   padding=self.padding,
                   strides=self.strides,
                   kernel_initializer=self.kernel_initializer,
                   activation="sigmoid",)(x)


        H = Multiply()([A,B])

        return H

    def __call__(self, x,*args, **kwargs):
        return self.call(x)
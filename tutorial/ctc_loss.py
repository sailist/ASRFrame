import numpy as np
import  keras.backend as K
import tensorflow as tf

a = [1,2,3,1,2,4,6,6,6,6]
b = [3,1,2,3,5,1,6,6,6,6]
c = [2,1,0,2,3,4,6,6,6,6]
y_true = np.stack([a,b,c])
y_pred = np.random.rand(3,15,7).astype(np.float32)


input_length = np.stack([[7],[8],[9]])

label_length = np.stack([[4],[4],[4]])

result = K.ctc_batch_cost(y_true,y_pred,input_length,label_length)
print(K.eval(result))
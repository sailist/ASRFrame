'''
作者：挥挥洒洒
来源：CSDN
原文：https://blog.csdn.net/u010420283/article/details/80303231
版权声明：本文为博主原创文章，转载请附上博文链接！
'''

# 暂时没有使用

from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt




def main():
    model = load_model('data/checkpoints/inception.026-1.07.hdf5')  # replaced by your model name
    # Get all our test images.
    image = 'v_ApplyLipstick_g01_c01-0105.jpg'
    images = plt.imread('v_ApplyLipstick_g01_c01-0105.jpg')
    plt.imshow("Image", images)
    # Turn the image into an array.

    # 设置可视化的层
    layer_1 = K.function([model.layers[0].input], [model.layers[1].output])
    f1 = layer_1([image_arr])[0]
    for _ in range(32):
        show_img = f1[:, :, :, _]
        show_img.shape = [149, 149]
        plt.subplot(4, 8, _ + 1)
        plt.subplot(4, 8, _ + 1)
        plt.imshow(show_img, cmap='gray')
        plt.axis('off')
    plt.show()
    # conv layer: 299
    layer_1 = K.function([model.layers[0].input], [model.layers[299].output])
    f1 = layer_1([image_arr])[0]
    for _ in range(81):
        show_img = f1[:, :, :, _]
        show_img.shape = [8, 8]
        plt.subplot(9, 9, _ + 1)
        plt.imshow(show_img, cmap='gray')
        plt.axis('off')
    plt.show()
    print('This is the end !')


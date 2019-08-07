from util.scripts import err_count

'''用于控制GPU'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""#不适用GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"#使用一个GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"#使用0/1两个GPU

err_count()
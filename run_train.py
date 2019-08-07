import config
from util.reader import ST_CMDS,Thchs30,Primewords,AiShell,Z200,TextDataGenerator

from acoustic.ABCDNN import DCBNN1D,DCBNN1Dplus
from acoustic.MAXM import MPBCONM,MPCONM,MCONM
from acoustic.WAVE import WAVEM

from language.DCNN import DCNN1D
from language.SOMM import SOMMalpha,SOMMword

'''数据集加载，注意使用前在config前配置好相应路径'''
stcmd = ST_CMDS(config.stcmd_datapath) # 据说还可以
thchs = Thchs30(config.thu_datapath) # 同质性太高，不过好拟合，可以用来测试模型的效果，在这个数据上都没法得到比较好的结果的就没啥使用的必要了
prime = Primewords(config.prime_datapath)
aishell = AiShell(config.aishell_datapath) # 据说数据集很差，不用该数据训练
z200 = Z200(config.z200_datapath)
wiki = TextDataGenerator(config.wiki_datapath)


'''用于控制GPU'''
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""#不适用GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#使用一个GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"#使用0/1两个GPU


'''语言模型——————————————————————————————————————————————————————————————————————————————————'''

'''基本没什么效果，卷着卷着就卷没了'''
# DCNN1D.train([thchs],None)

'''效果目前来看很不错，但是目前（2019年7月9日）下语料不足，貌似过拟合了，需要扩充语料后再尝试'''
# SOMMalpha.train(wiki,None)
SOMMalpha.train(wiki,os.path.join(config.language_model_dir,"SOMMalpha_epoch_327_step_163500.h5"))

# SOMMword.train([thchs],None) # 注意SOMMword的train方法版本有点旧


'''声学模型——————————————————————————————————————————————————————————————————————————————————'''
# MCONM.train([thchs],config.join_model_path("./acoustic/MCONM_epoch_55_step_55000.h5"))

# MPCONM.train([thchs,stcmd,prime,z200],config.join_model_path("./acoustic/MPCONM_epoch_313_step_313000.h5"))
# WAVEM.train([thchs],)

'''目前最有效的模型'''
# DCBNN1D.train([thchs],epoch=140)
# DCBNN1D.train([thchs,stcmd,prime,aishell,z200],load_model=config.join_model_path("./acoustic/DCBNN1D_epoch_490_step_490000.h5"))


'''2019年7月2日08:34:19，开始尝试'''
'''目前来看效果反而没有dcbnn1d好，如果添加残差结构可能会好一些'''
# DCBNN1Dplus.train([thchs,z200,stcmd,aishell,prime],load_model=None)

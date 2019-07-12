import config
from examples import train_language_model as lexample
from examples import train_acoustic_model as aexample
from util.reader import ST_CMDS,Thchs30,Primewords,AiShell,Z200,TextDataGenerator

stcmd = ST_CMDS(config.stcmd_datapath) # 据说还可以
thchs = Thchs30(config.thu_datapath) # 同质性太高，不过好拟合，可以用来测试模型的效果，在这个数据上都没法得到比较好的结果的就没啥使用的必要了
prime = Primewords(config.prime_datapath)
aishell = AiShell(config.aishell_datapath) # 据说数据集很差，不用该数据训练
z200 = Z200(config.z200_datapath)
wiki = TextDataGenerator(config.wiki_datapath)

config.model_dir = "./model/"

'''语言模型——————————————————————————————————————————————————————————————————————————————————'''

'''基本没什么效果，卷着卷着就卷没了'''
# lexample.train_dcnn1d([thchs],load_model=None)

'''效果目前来看很不错，但是目前（2019年7月9日）下语料不足，貌似过拟合了，需要扩充语料后再尝试'''
# lexample.train_somiao([thchs,stcmd,prime,aishell,z200],load_model=None)
# lexample.train_sommalpha(wiki, load_model=None)
# lexample.train_sommalpha(wiki, load_model=config.join_model_path("./language/SOMMalpha_step_50500.h5"))


'''声学模型——————————————————————————————————————————————————————————————————————————————————'''

'''目前最有效的模型'''
# aexample.train_dcbnn1d([z200], config.join_model_path("./acoustic/SOMMalpha_step_45500.h5"))
# aexample.train_dcbnn1d([thchs,z200,prime,aishell,stcmd]
#                   ,load_model=config.join_path("DRModel_step_45000.h5"))


'''2019年7月2日08:34:19，开始尝试'''
'''目前来看效果反而没有dcbnn1d好，如果添加残差结构可能会好一些'''
aexample.train_dcbnn1dplus([thchs,z200,stcmd,aishell,prime],
                           load_model=None)


'''2019年7月2日08:34:14，效果不好，停止训练，具体情报参考类下的注释'''
# aexample.summary_dcbann1d([thchs],
#                         load_model=config.join_path("DRAModel_step_7000.h5"))



'''======================================================================================'''
'''以下训练方法停止更新，效果都不怎么好=_='''
# examples.train_trans("/home/sailist/download/")
# examples.try_predict_trans("/data/voicerec/z200","./model/model_20000.h5")
# examples.train_trans("/data/voicerec/z200",)
# examples.train_trans("/data/voicerec/z200","./model/model_5000.h5")
# examples.train_trans_thu("/data/voicerec/dataset/dataset/data_thchs30/")

'''效果没有dr好，但勉强也还可以，可以不考虑了'''
# examples.train_dcnn2d([thchs,stcmd],)
# examples.train_dcnn2d(datagenes  = [thchs],
#                        load_model = config.join_path("model_50500.h5"))

'''训练废了，大数据集没法拟合，在小数据集上效果还行，大概是100左右数据集上1%，两三百左右数据集上5%...主要训练太慢没有精力继续训练了。'''
# examples.train_las("/data/voicerec/dataset/dataset/data_thchs30/")
# examples.train_las("/data/voicerec/dataset/dataset/data_thchs30/","./model/LASModel_model_10800.h5")

'''ReLASModel模型非常辣鸡,不过这是我写的=_='''
# examples.train_relas(config.thu_datapath)



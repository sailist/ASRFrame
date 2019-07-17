# 用于测试数据集是否能全部跑通

import config
from util.reader import VoiceLoader,VoiceDatasetList
from util.reader import Thchs30,Z200,AiShell,Primewords,ST_CMDS
from util.mapmap import PinyinMapper
from feature.mel_feature import MelFeature5


stcmd = ST_CMDS(config.stcmd_datapath) # 据说还可以
thchs = Thchs30(config.thu_datapath) # 同质性太高，不过好拟合，可以用来测试模型的效果，在这个数据上都没法得到比较好的结果的就没啥使用的必要了
prime = Primewords(config.prime_datapath)
aishell = AiShell(config.aishell_datapath) # 据说数据集很差，不用该数据训练
z200 = Z200(config.z200_datapath)


# datagenes = [thchs,stcmd,prime,aishell,z200]
datagenes = [thchs]
if __name__ == "__main__":
    w, h = 1600, 200
    max_label_len = 64
    batch_size = 16

    dataset = VoiceDatasetList()
    x_set, y_set = dataset.merge_load(datagenes)
    pymap = PinyinMapper(sil_mode=-1)
    vloader = VoiceLoader(x_set, y_set,
                          batch_size=16,
                          feature_pad_len=w,
                          n_mels=h,
                          max_label_len=max_label_len,
                          pymap=pymap,
                          divide_feature_len=8,
                          melf=MelFeature5(),
                          all_train=True
                          )

    viter = vloader.create_iter(one_batch=True)
    for i,_ in enumerate(viter):
        print(f"\r{i*batch_size}.",end="\0",flush=True)
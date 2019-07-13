import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import config
from examples.model_summary import summary_dcbann1d

from util.reader import ST_CMDS,Thchs30,Primewords,AiShell,Z200
stcmd = ST_CMDS(config.stcmd_datapath) # 据说还可以
thchs = Thchs30(config.thu_datapath) # 同质性太高
prime = Primewords(config.prime_datapath)
aishell = AiShell(config.aishell_datapath) # 据说数据集很差，不用该数据训练
z200 = Z200(config.z200_datapath)


if __name__ == "__main__":
    summary_dcbann1d([stcmd], config.join_model_path("./acoustic/DCBNN1D_cur_best.h5"))

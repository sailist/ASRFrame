'''
用于清洗数据，只需要传入相应数据的根目录即可
'''
import config
from util.cleaner import Thchs30,Z200,AiShell,Primewords,ST_CMDS

if __name__ == "__main__":
    # Thchs30(config.thu_datapath) # 清华的数据集不需要清洗，代码过时了
    Z200(config.z200_datapath).clean()
    AiShell(config.aishell_datapath).clean()
    Primewords(config.prime_datapath).clean()
    ST_CMDS(config.stcmd_datapath).clean()

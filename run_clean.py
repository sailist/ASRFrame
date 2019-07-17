'''
用于清洗数据，只需要传入相应数据的根目录即可
'''
import config
from util.dataset import Thchs30,Z200,AiShell,Primewords,ST_CMDS

if __name__ == "__main__":
    Thchs30(config.thu_datapath).label_dataset()
    Z200(config.z200_datapath).label_dataset()
    AiShell(config.aishell_datapath).label_dataset()
    Primewords(config.prime_datapath).label_dataset()
    ST_CMDS(config.stcmd_datapath).label_dataset()

    Thchs30(config.thu_datapath).clean_dataset()
    Z200(config.z200_datapath).clean_dataset()
    AiShell(config.aishell_datapath).clean_dataset()
    Primewords(config.prime_datapath).clean_dataset()
    ST_CMDS(config.stcmd_datapath).clean_dataset()
'''
用于清洗数据，只需要传入相应数据的根目录即可
'''
import config
from examples.dataset_clean import *

if __name__ == "__main__":
    # clean_thu(config.thu_datapath) # 清华的数据集不需要清洗，代码过时了
    clean_z200(config.z200_datapath)
    clean_aishell(config.aishell_datapath)
    clean_prime(config.prime_datapath)
    clean_stcmds(config.stcmd_datapath)

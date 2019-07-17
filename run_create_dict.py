'''
用于清洗数据，只需要传入相应数据的根目录即可
'''
import config
import os
# from util.cleaner import Thchs30,Z200,AiShell,Primewords,ST_CMDS
from util.dataset import Thchs30,Z200,AiShell,Primewords,ST_CMDS
from util.dataset import Datautil
if __name__ == "__main__":
    Thchs30(config.thu_datapath).initial() # 清华的数据集不需要清洗，代码过时了
    Z200(config.z200_datapath).initial()
    AiShell(config.aishell_datapath).initial()
    Primewords(config.prime_datapath).initial()
    ST_CMDS(config.stcmd_datapath).initial()

    dict_path = os.path.join(config.project_path,"util","dicts")

    Datautil.merge_dict([config.thu_datapath,
                         config.z200_datapath,
                         config.prime_datapath,
                         config.aishell_datapath,
                         config.stcmd_datapath],
                        output_dir_path=dict_path)

    Datautil.filter_dict(os.path.join(dict_path,"chs_dict.dict"),
                         os.path.join(dict_path,"filter_chs_dict.dict"))
    Datautil.filter_dict(os.path.join(dict_path,"py_dict.dict"),
                         os.path.join(dict_path,"filter_py_dict.dict"))


    '''然后可以自行查看一下字典，合并字典，设置字典目录，然后运行下面的代码'''

    # Thchs30(config.thu_datapath).clean_dataset()
    # Z200(config.z200_datapath).clean_dataset()
    # AiShell(config.aishell_datapath).clean_dataset()
    # Primewords(config.prime_datapath).clean_dataset()
    # ST_CMDS(config.stcmd_datapath).clean_dataset()
# 用于统计数据的脚本，部分脚本运行时间可能会很长
from util.reader import *

def check_path(path):
    return path is not None and os.path.exists(path)

def summary(gene:VoiceDataGenerator):
    x_set, y_set = gene.load_from_path()
    vloader = VoiceLoader(x_set, y_set,vad_cut = False,check=False)
    print(f"start to summary the {gene.__class__.__name__} dataset")
    vloader.summery(audio=True,
                    label=True,
                    plot=True,
                    dataset_name = gene.__class__.__name__)

def summary_z200(path):
    if not check_path(path):
        print(f"{path} not available.")
        return
    summary(Z200(path))

def summary_thchs30(path):
    if not check_path(path):
        print(f"{path} not available.")
        return
    summary(Thchs30(path))

def summary_stcmds(path):
    if not check_path(path):
        print(f"{path} not available.")
        return
    summary(ST_CMDS(path))

def summary_aishell(path):
    if not check_path(path):
        print(f"{path} not available.")
        return
    summary(AiShell(path))

def summary_prime(path):
    if not check_path(path):
        print(f"{path} not available.")
        return
    summary(Primewords(path))
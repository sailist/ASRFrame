import platform
import os

max_feature_time_stamp = 256
max_label_len = 30
default_sample = 16000

thu_datapath = None
z200_datapath = None
aishell_datapath = None
prime_datapath = None
stcmd_datapath = None

wiki_datapath = None

if platform.system() == "Linux":
    thu_datapath = "/data/voicerec/dataset/dataset/thchs30-openslr/data_thchs30"
    z200_datapath = "/data/voicerec/z200"
    aishell_datapath = "/data/voicerec/ALShell-1/data_aishell"
    prime_datapath = "/data/voicerec/Primewords Chinese Corpus Set 1/primewords_md_2018_set1"
    stcmd_datapath = "/data/voicerec/Free ST Chinese Mandarin Corpus/ST-CMDS-20170001_1-OS"
    wiki_datapath = "/data/voicerec/wiki/wiki_corpus_2"
elif platform.system() == "Windows":
    thu_datapath = r"C:\E\jupyter_notebook\voice_reco\Dataset\thchs30"
    z200_datapath = r"C:\E\jupyter_notebook\voice_reco\Dataset\z200"
    aishell_datapath = r"C:\E\jupyter_notebook\voice_reco\Dataset\data_aishell"
    prime_datapath = r"C:\E\jupyter_notebook\voice_reco\Dataset\primewords_md_2018_set1"
    stcmd_datapath = r"C:\E\jupyter_notebook\voice_reco\Dataset\ST-CMDS-20170001_1-OS"

model_dir = "./model/" #模型存储或读取的路径，注意所有模型默认存储都按这个路径来，但也可以手动设置
acoustic_model_dir = os.path.join(model_dir, "acoustic")
language_model_dir = os.path.join(model_dir, "language")

loss_dir = "./loss_plot/"
acoustic_loss_dir = os.path.join(loss_dir,"acoustic")
language_loss_dir = os.path.join(loss_dir,"language")

join_model_path = lambda x:os.path.join(model_dir, x)


latest = {
    "trans2d":"TransModel2D_step_38000.h5",
    "las":"LASModel_model_10800.h5",
    "dcbnn1d":"DCBNN1D_cur_best.h5",
}
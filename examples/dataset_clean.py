# 用于清洗数据的可以直接运行的脚本

from util.cleaner import *
def check_path(path):
    return path is not None and os.path.exists(path)

def clean_thu(path):
    '''
    thchs30数据集
    传入包含data/dev/Im_phone...的目录路径
    :param path:
    :return:
    '''
    thu = Thchs30(path)
    thu.clear_npfile()
    # thu.gene_pinyin()

def clean_z200(path):
    '''
    Aidatatang_200zh数据集
    传入包含G0002/G0003...的路径
    :param path:
    :return:
    '''
    z200 = Z200(path)
    z200.delete_number_file()
    z200.clear_npfile()
    z200.gene_pinyin()

def clean_aishell(path):
    '''
    AISHELL数据集
    传入包含dev/test/train/...的路径
    :param path:
    :return:
    '''
    als = AiShell(path)
    als.gene_pinyin()

def clean_prime(path):
    '''
    Primewords Chinese Corpus Set 1数据集
    传入包含audio_files/set1_transcript.json 的路径
    :param path:
    :return:
    '''
    prime = Primewords(path)
    prime.gene_pinyin()

def clean_stcmds(path):
    '''
    ST-CMDS数据集
    传入包含*.wav/*.txt/*.metadata的路径
    :param path:
    :return:
    '''
    stcmds = ST_CMDS(path)
    stcmds.gene_pinyin()
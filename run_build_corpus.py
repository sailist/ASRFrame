# 用于创建修正错误拼音的模型，使用kenLM
from util.build_corpus import build_py_corpus


if __name__ == "__main__":
    import config
    from util.reader import Z200,ST_CMDS,Primewords,Thchs30,AiShell,VoiceDatasetList

    z200 = Z200(config.z200_datapath)
    std = ST_CMDS(config.stcmd_datapath)
    prime = Primewords(config.prime_datapath)
    ais = AiShell(config.aishell_datapath)
    thu = Thchs30(config.thu_datapath)

    datagene = VoiceDatasetList()
    xs,ys = datagene.merge_load([z200,std,prime,ais,thu])
    build_py_corpus(ys,"pylm_corpus.txt")
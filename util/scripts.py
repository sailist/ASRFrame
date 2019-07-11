from util.reader import Z200,Thchs30,Primewords,ST_CMDS,AiShell
import os
import config

'''!"'(),.?@q~“”…　、。」！（），？Ａａｂｃｋｔ'''

def count_chs():
    z200 = Z200(config.z200_datapath)
    thchs = Thchs30(config.thu_datapath)
    prime = Primewords(config.prime_datapath)
    stcmd = ST_CMDS(config.stcmd_datapath)
    aishell = AiShell(config.aishell_datapath)

    lst = [z200,thchs,prime,stcmd,aishell]

    chs_set = set()
    for i in lst:
        _,y_set = i.load_from_path(choose_x=False,choose_y=True)

        for j,yfs in enumerate(y_set):
            print(f"\r{j},{yfs}",sep="\0",flush=True)
            with open(yfs,encoding="utf-8") as f:
                line = f.readline().strip()
                chs_set.update(line)

    save_path = os.path.abspath("./dataset_chs.txt")


    with open(save_path,"w",encoding="utf-8") as w:
        for i in chs_set:
            w.write(f"{i}")
    print(f"dict has been saved in {save_path}.")


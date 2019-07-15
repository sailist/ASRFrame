import config
from util.reader import Z200,Thchs30,Primewords,ST_CMDS,AiShell


if __name__ == "__main__":
    Thchs30(config.thu_datapath).summary()
    AiShell(config.aishell_datapath).summary()
    Primewords(config.prime_datapath).summary()
    ST_CMDS(config.stcmd_datapath).summary()
    Z200(config.z200_datapath).summary()


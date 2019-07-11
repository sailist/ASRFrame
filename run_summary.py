import examples
import config


if __name__ == "__main__":
    examples.summary_thchs30(config.thu_datapath)
    examples.summary_aishell(config.aishell_datapath)
    examples.summary_prime(config.prime_datapath)
    examples.summary_stcmds(config.stcmd_datapath)
    examples.summary_z200(config.z200_datapath)

    pass
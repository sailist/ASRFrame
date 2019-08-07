import config
from jointly.DCSOM import DCSOM
from language.SOMM import SOMMalpha
# dcbnn = config.join_model_path("DCBNN1D_cur_best.h5")
dcbnn = config.join_model_path("DCBNN1D_step_30000.h5")
sommalpha = config.join_model_path("language/SOMMalpha_step_11500.h5")


if __name__ == "__main__":
    DCSOM.record_from_wav("path/to/wav") # 要求采样率必须为16000
    DCSOM.real_predict(dcbnn,sommalpha)
    SOMMalpha.real_predict(sommalpha)
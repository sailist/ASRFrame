from examples.real_predict import predict_sommalpha,predict_dchmm,predict_dcsom
import config
# dcbnn = config.join_model_path("DCBNN1D_cur_best.h5")
dcbnn = config.join_model_path("DCBNN1D_step_30000.h5")
sommalpha = config.join_model_path("language/SOMMalpha_step_11500.h5")


if __name__ == "__main__":
    predict_dcsom(dcbnn,sommalpha)
    # predict_sommalpha(sommalpha)
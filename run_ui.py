from visualization.voicev import Front
from jointly.DCSOM import DCSOM
from util.mapmap import PinyinMapper,ChsMapper

if __name__ == "__main__":
    dcs = DCSOM(acmodel_input_shape=(1600, 200),
                acmodel_output_shape=(200,),
                lgmodel_input_shape=(200,),
                py_map=PinyinMapper(sil_mode=-1),
                chs_map=ChsMapper(),
                divide_feature=8)

    # dcs.compile("../model/DCBNN1D_step_326000.h5",
    #             "../model/language/SOMMalpha_step_18000.h5")
    dcs.compile("./model/DCBNN1D_cur_best.h5",
                "./model/language/SOMMalpha_step_18000.h5")
    front = Front()
    front.create(dcs)


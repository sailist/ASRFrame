# 用来统计模型经常会识别错误的拼音有哪些，这要求模型可以识别错，但拼音与说话的长度应该是一一对应的，不是一一对应的拼音会被跳过
# TODO 但是根据我简单的测试，貌似长度一样的识别正确率都是100%，识别错误的长度都不一样...???这是什么模型操作
from acoustic import *
from util.reader import *
from feature.mel_feature import MelFeature5
from util.mapmap import PinyinMapper

def summary_dcbann1d(datagenes:list, load_model = None):
    w, h = 1600, 200
    max_label_len = 64

    dataset = VoiceDatasetList()
    x_set, y_set = dataset.merge_load(datagenes)
    pymap = PinyinMapper(sil_mode=-1)
    vloader = VoiceLoader(x_set, y_set,
                          batch_size= 16,
                          feature_pad_len = w,
                          n_mels=h,
                          max_label_len=max_label_len,
                          pymap=pymap,
                          melf=MelFeature5(),
                          divide_feature_len=8,
                          # cut_sub=64,
                          )

    model_helper = DCBNN1D(pymap)
    model_helper.compile(feature_shape=(w, h), label_max_string_length=max_label_len, ms_output_size=pymap.max_index+1)

    if load_model is not None:
        load_model = os.path.abspath(load_model)
        model_helper.load(load_model)


    viter = vloader.create_iter(one_batch=True)
    all_err_dict = {}
    with open("./error_dict.txt", "w", encoding="utf-8") as w:
        for batch in viter:
            test_res = model_helper.test(batch,use_ctc=True,return_result=True)
            err_dict = test_res["err_pylist"]
            for k,lst in err_dict.items():
                errlist = all_err_dict.setdefault(k,[])
                errlist.extend(lst)

            for k,v in err_dict.items():
                v = set(v)
                w.write(f"{k},{' '.join(v)}")
    print(all_err_dict)

    with open("./error_dict.txt", "w", encoding="utf-8") as w:
        for k,v in all_err_dict.items():
            v = set(v)
            w.write(f"{k},{' '.join(v)}")








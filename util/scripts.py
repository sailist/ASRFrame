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

def err_count():
    '''
    根据音频统计错误数据集
    :return:
    '''
    z200 = Z200(config.z200_datapath)
    thchs = Thchs30(config.thu_datapath)
    prime = Primewords(config.prime_datapath)
    stcmd = ST_CMDS(config.stcmd_datapath)
    aishell = AiShell(config.aishell_datapath)

    lst = [z200, thchs, prime, stcmd, aishell]

    from acoustic.ABCDNN import DCBNN1D
    from util.reader import PinyinMapper,VoiceDatasetList,VoiceLoader
    from feature.mel_feature import MelFeature5

    w, h = 1600, 200
    max_label_len = 64
    pymap = PinyinMapper(sil_mode=-1)

    model_helper = DCBNN1D(pymap)
    model_helper.compile(feature_shape=(w, h), label_max_string_length=max_label_len,
                         ms_output_size=pymap.max_index + 1)

    model_helper.load(os.path.join(config.model_dir,"cur_best_DCBNN1D_epoch_722_step_722000.h5"))


    dataset = VoiceDatasetList()
    x_set, y_set = dataset.merge_load(lst)

    vloader = VoiceLoader(x_set, y_set,
                          batch_size=16,
                          feature_pad_len=w,
                          n_mels=h,
                          max_label_len=max_label_len,
                          pymap=pymap,
                          melf=MelFeature5(),
                          divide_feature_len=8,
                          all_train=True,
                          )
    viter = vloader.create_iter(one_batch = True,return_word=True)

    with open("data_err.txt","w",encoding="utf-8") as w:
        for i,batch in enumerate(viter):
            [_, ys, _, label_len], words = batch
            py_true_b = pymap.batch_vector2pylist(ys,return_word_list=True,return_list=True)
            py_pred_b = model_helper.predict(batch)
            for py_true,py_pred,llen,word in zip(py_true_b,py_pred_b,label_len,words):
                llen = llen[0]
                # w.write(f"{pyt}")
                py_true = py_true[:llen]

                py_true = " ".join(py_true)
                py_pred = " ".join(py_pred)

                print(f"\r——{i*16}.",end="\0",flush=True)
                # print(word.strip(),py_pred)
                w.write(f"{word.strip()}\t{py_true}\t{py_pred}\n")




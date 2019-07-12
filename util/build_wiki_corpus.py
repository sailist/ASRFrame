# 专门写的用于切分wiki语料的文件，主方法是build

import os
import json
import thulac
from util.number_convert import convert
import re
from pypinyin import pinyin,Style

re_num = re.compile("([0-9]+)")
re_alpha = re.compile("([a-z]+)")
pypinyin = lambda word:pinyin(word,Style.TONE3,errors=lambda x:len(x))
model = thulac.thulac()

def filtew(w, n):
    w = w.strip("\xa0")
    if n == "w":
        return True

    if len(w) == 0:
        return True
    if n in ["", "x", "g"]:
        return False
    if n != "m" and w.isdigit():
        return False
    return True


def translate(w, n):
    if n == "w":
        return "\n"

    if n not in ["m", "t"]:
        return w
    if w.isdigit():
        #         print(w)
        w = convert(w, False)
        return w

    digit_list = re.findall(re_num, w)
    for digit in digit_list:
        #         if random.random()>0.5:
        #             chdigit = convert(digit,False)
        #         else:

        chdigit = convert(digit, True)
        #         print(digit,chdigit)
        w = w.replace(digit, chdigit)
    return w

def create_pinyin_for_seg_word(word):
    pylist = pypinyin(word)
    stri = 0
    pyi = 0
    ls = []
    pyls = []
    size = len(pylist)
    while pyi<size:
        if isinstance(pylist[pyi][0],int):
            stri += pylist[pyi][0]
            pyi+=1
            continue
        else:
            ls.append(word[stri])
            pyls.append(pylist[pyi][0])
            stri+=1
            pyi+=1
    return ls,pyls

def create_pinyin_for_seg(word_list,thresh=7):
    '''

    :param word_list:
    :param thresh: 句子上限长度，不绝对，可能会超过1-3个
    :return:
    '''
    all_ls = []
    all_pyls = []
    num = 0
    for word in word_list:
        if not isinstance(word,str):
            print(word)
        word = word.strip()
        if len(word) == 0:
            continue
        ls,pyls = create_pinyin_for_seg_word(word)
        if (len(ls) == 0 and len(all_ls) != 0) or num>thresh:
            yield all_ls,all_pyls
            num = 0
            all_ls = []
            all_pyls = []
            continue
        num+=len(ls)
        all_ls.append("".join(ls))
        all_pyls.append(" ".join(pyls))
    if len(all_ls) != 0:
        yield all_ls,all_pyls



def cut_wikifile(fn):
    # all_cut = []
    with open(fn,encoding="utf-8") as f:
        for line in f:
            jstr = json.loads(line)
            string = jstr["text"]
            stringlist = string.replace("。","\n").split("\n")
            for string in stringlist:
                cut_res = model.cut(string)
                # all_cut.extend(cut_res)
                yield cut_res



def build(root_path, output_dir):
    '''
    将wiki语料中的title提取出来，分词后用拼音标注，并按 汉字[tab]拼音的方式存储
    :param root_path: wiki 语料的根目录
    :param output_dir: 输出的目录，如果不存在会直接建立
    :return:
    '''
    os.makedirs(output_dir, exist_ok=True)
    fs = os.listdir(root_path)
    fs = [os.path.join(root_path, f) for f in fs]

    wikifs = []
    for f in fs:
        wikif = os.listdir(f)
        wikif = [os.path.join(f, wi) for wi in wikif]
        wikifs.extend(wikif)
    count = 0
    for i, fn in enumerate(wikifs):
        if i < 6:
            continue
        cut_iter = cut_wikifile(fn)
        fpath = os.path.join(output_dir, f"{i}.txt")
        with open(fpath, "w", encoding="utf-8") as w:
            for j, cut_line in enumerate(cut_iter):
                try:
                    cut_line = [translate(w, pos) for w, pos in cut_line if filtew(w, pos)]
                except:
                    continue
                ls_iter = create_pinyin_for_seg(cut_line,7) # 按长度为7进行切分
                for k, (all_ls, all_pyls) in enumerate(ls_iter):
                    w.write(f"{''.join(all_ls)}\t{' '.join(all_pyls)}\n")
                    # w.write(f"{")
                    count += 1
                print(f"\r{fpath}:{count}:{j}_{k}", end="\0", flush=True)


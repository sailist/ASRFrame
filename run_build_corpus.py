# 【2019年7月17日】该清洗方法用于清洗wiki的语料，但目前不保证正确
# 该清洗方法可能会持续很长时间，我当初跑了近两天也还没有跑完，全部跑完应该有近两千万的样本
from util.build_wiki_corpus import build

if __name__ == "__main__":
    build("/data/voicerec/wiki/wiki_zh","/data/voicerec/wiki/wiki_corpus_2")
# 该清洗方法可能会持续很长时间，我当初跑了近两天也还没有跑完，全部跑完应该有近两千万的样本
from util.build_wiki_corpus import build

if __name__ == "__main__":
    build("path/of/root","path/of/output")
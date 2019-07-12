from language.Pinyin2Hanzi import DefaultHmmParams
from language.Pinyin2Hanzi import all_pinyin
from language.Pinyin2Hanzi import viterbi

all_pinyin = list(all_pinyin())

class PHHMM():
    '''传统的HMM模型，使用Pinyin2Hanzi库，位于该目录下（未包含训练用代码）'''
    def __init__(self):
        self.hmmparams = DefaultHmmParams()

    def compile(self):
        pass

    def predict(self,pylist):
        '''TODO 这里的pylist的格式没有规范'''
        pylist = [i.replace("ue","ve") for i in pylist]
        pylist = [i.strip("12345") for i in pylist if i.strip("12345") in all_pinyin]

        if len(pylist) == 0:
            return [],0
        result = viterbi(hmm_params=self.hmmparams, observations=pylist, path_num=1)

        for item in result:
            return item.path,item.score

    def load(self,*args):
        pass

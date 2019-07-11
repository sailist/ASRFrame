import distance

class EvaluateDistance():
    def compare_sent(self,sent_true:str,sent_pred:str):
        '''

        :param sent_true:
        :param sent_pred:
        :return: int value
        '''
        return distance.levenshtein(sent_true,sent_pred),distance.levenshtein(sent_true,sent_pred,normalized=True)

    def batch_compare_sent(self,y_true,y_pred,return_every = False):
        count_list = []
        for sent_true,sent_pred in zip(y_true,y_pred):
            count_list.append(self.compare_sent(sent_true,sent_pred))

        if return_every:
            return count_list
        else:
            return sum(count_list)



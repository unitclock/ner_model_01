from torch import nn
from transformers import BertTokenizer,BertModel

class NerModel(nn.Module):
    def __init__(self,pretained_path:str,dropout:float):
        super(NerModel, self).__init__()
        self.__roberta = BertModel.from_pretrained(pretained_path) 
        return
        
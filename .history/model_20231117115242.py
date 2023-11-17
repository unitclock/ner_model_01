from torch import nn
from transformers import BertTokenizer,BertModel,BertConfig

class NerModel(nn.Module):
    def __init__(self,pretained_path:str,hidden_dim:int,dropout:float,targets:dict):
        super(NerModel, self).__init__()
        self.__bert = BertModel.from_pretrained(pretained_path)
        self.__bert_config = BertConfig.from_pretrained(pretained_path)
        self.__hidden_size = self.__bert_config.hidden_size
        self.__classifier = nn.Linear(self.__hidden_size,len(targets))
        return
        
    def forward(self):
        pass
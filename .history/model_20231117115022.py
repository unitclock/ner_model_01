from torch import nn
from transformers import BertTokenizer,BertModel,BertConfig

class NerModel(nn.Module):
    def __init__(self,pretained_path:str,hidden_dim:int,dropout:float,targets:dict):
        super(NerModel, self).__init__()
        self.__roberta = BertModel.from_pretrained(pretained_path)
        self.__hidden_size = BertConfig.from_pretrained(pretained_path) 
        self.__classifier = nn.Linear()
        return
        
    def forward(self):
        pass
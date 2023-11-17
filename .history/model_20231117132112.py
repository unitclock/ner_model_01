from torch import nn
import torch
from transformers import BertTokenizer,BertModel,BertConfig

class NerModel(nn.Module):
    def __init__(self,pretained_path:str,dropout:float,targets:dict):
        super(NerModel, self).__init__()
        self.__bert = BertModel.from_pretrained(pretained_path)
        self.__bert_config = BertConfig.from_pretrained(pretained_path)
        self.__hidden_size = self.__bert_config.hidden_size
        self.__dropout = nn.Dropout(dropout)
        self.__classifier = nn.Linear(in_features=self.__hidden_size,out_features=len(targets),bias=True)
        return
        
    def forward(self,input_ids,):
        with torch.no_grad():
            
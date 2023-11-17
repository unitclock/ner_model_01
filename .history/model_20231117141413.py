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
        self.__softmax = nn.Softmax(dim =1)
        return
        
    def forward(self,input_ids,attention_mask):
        token_series = self.bert(input_ids,attention_mask)[0]
        token_series = self.__dropout(token_series)
        output = 
        


        
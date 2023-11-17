from torch import nn
import torch
from transformers import BertTokenizerFast,BertModel,BertConfig
import torch.nn.functional as F

class NerModel(nn.Module):
    bio_unique_labels =[
        'B-dis', 'B-sym', 'B-pro', 'B-equ', 'B-dru', 
        'B-ite', 'B-bod', 'B-dep', 'B-mic', 'I-dis', 
        'I-sym', 'I-pro', 'I-equ', 'I-dru', 'I-ite', 
        'I-bod', 'I-dep', 'I-mic', 'O-dis', 'O-sym', 
        'O-pro', 'O-equ', 'O-dru', 'O-ite', 'O-bod', 
        'O-dep', 'O-mic', 'E-dis', 'E-sym', 'E-pro', 
        'E-equ', 'E-dru', 'E-ite', 'E-bod', 'E-dep', 
        'E-mic', 'S-dis', 'S-sym', 'S-pro', 'S-equ', 
        'S-dru', 'S-ite', 'S-bod', 'S-dep', 'S-mic', 
        'O']
    def __init__(self,dropout= 0.1):
        super(NerModel, self).__init__()
        self.__pretained_path = "./premodels/bert-base-chinese"
        self.__bert = BertModel.from_pretrained(self.__pretained_path)
        self.__bert_config = BertConfig.from_pretrained(self.__pretained_path)
        self.__hidden_size = self.__bert.config.hidden_size
        self.__dropout = nn.Dropout(dropout)
        self.__classifier = nn.Linear(in_features=self.__hidden_size,out_features=len(self.bio_unique_labels),bias=True)
        self.__softmax = nn.Softmax(dim =1)
        return
        
    def forward(self,input_ids,attention_mask):
        token_series = self.__bert(input_ids,attention_mask)[0]
        token_series = self.__dropout(token_series)
        token_series = self.__classifier(token_series)
        probabilities = F.softmax(token_series,dim=1) 
        return probabilities, (a,b,c)
        



        
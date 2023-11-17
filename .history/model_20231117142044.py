from torch import nn
import torch
from transformers import BertTokenizerFast,BertModel,BertConfig

class NerModel(nn.Module):
    def __init__(self,dropout= 0.1):
        self.__pretained_path = "./bert-base-chinese"
        super(NerModel, self).__init__()
        self.__bert = BertModel.from_pretrained(self.__pretained_path)
        self.__bert_config = BertConfig.from_pretrained(self.__pretained_path)
        self.__hidden_size = self.__bert_config.hidden_size
        self.__dropout = nn.Dropout(dropout)
        self.__classifier = nn.Linear(in_features=self.__hidden_size,out_features=len(targets),bias=True)
        self.__softmax = nn.Softmax(dim =1)
        return
        
    def forward(self,input_ids,attention_mask):
        token_series = self.bert(input_ids,attention_mask)[0]
        token_series = self.__dropout(token_series)
        output = self.__softmax(token_series)
        return output
        

if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained("./bert-base-chinese/")
    t = tokenizer(["人生路漫漫","明日何其多"],return_tensors="pt")
    input_ids = t["input_ids"]
    print(type(input_ids),input_ids)
    token_type_ids = t["token_type_ids"]
    print(type(token_type_ids),token_type_ids)
    attention_mask = t["attention_mask"]
    print(type(attention_mask),attention_mask)

    model = NerModel(pretained_path="./b")

        
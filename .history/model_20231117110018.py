from torch import nn
from transformers import BertTokenizer,BertModel

class NerModel(nn.Module):
    def __init__(self):
        super(NerModel, self).__init__()
        pass
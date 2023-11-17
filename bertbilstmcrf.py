import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchcrf import CRF

class Bert_BiLSTM_CRF(nn.Module):
    def __init__(self, bert_path: str, tag_dict: dict, hidden_dim: int, dropout: float):
        super(Bert_BiLSTM_CRF, self).__init__()

        # BERT 模型
        self.bert = BertModel.from_pretrained(bert_path)
        self.hidden_dim = hidden_dim
        self.tagset_size = len(tag_dict)

        self.bilstm = nn.LSTM(input_size=self.bert.config.hidden_size,
                              hidden_size=hidden_dim // 2, num_layers=1, dropout=dropout, bidirectional=True, batch_first=True)

        self.fc = nn.Linear(hidden_dim, self.tagset_size)
        self.crf = CRF(num_tags=self.tagset_size, batch_first=True)

    def forward(self, input_ids, attention_mask=None, tags=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_output, _ = self.bilstm(bert_output[0])
        logits = self.fc(lstm_output)

        if tags is not None:
            loss = -1 * self.crf(logits, tags, mask=attention_mask.byte())
            return loss
        else:
            output = self.crf.decode(emissions=logits, mask=attention_mask.byte())
            return output
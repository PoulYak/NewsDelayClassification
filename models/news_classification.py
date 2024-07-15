from typing import Optional

import transformers
from torch import nn


class NewsClassification(nn.Module):
    def __init__(self, model_name: Optional[str]):
        super(NewsClassification, self).__init__()
        self.model_name = model_name
        self.bert = transformers.BertModel.from_pretrained(self.model_name)
        self.__freeze_some_layers()
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def __freeze_some_layers(self):
        """Freeze layers of Bert except last o"""
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        for layer in self.bert.encoder.layer[:10]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, output_1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                return_dict=False)
        output_2 = self.dropout(output_1)
        output_3 = self.classifier(output_2)
        output = self.sigmoid(output_3)

        return output

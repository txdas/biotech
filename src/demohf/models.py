from transformers import BertForSequenceClassification
from typing import List, Optional, Tuple, Union
import torch


class BertForRegression(BertForSequenceClassification):
    def __init__(self, config):
        super(BertForRegression, self).__init__(config)
        self.num_labels = 1

    def forward(self, input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None, ):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                            return_dict=return_dict)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))

        return (loss, logits) if loss is not None else logits

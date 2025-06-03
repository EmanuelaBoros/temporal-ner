from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel, AutoConfig, BertConfig
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union
import logging, json, os

from .configuration_stacked import ImpressoConfig

logger = logging.getLogger(__name__)


def get_info(label_map):
    num_token_labels_dict = {task: len(labels) for task, labels in label_map.items()}
    return num_token_labels_dict


class ExtendedMultitaskModelForTokenClassification(PreTrainedModel):

    config_class = ImpressoConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_token_labels_dict = get_info(config.label_map)
        self.config = config

        self.bert = AutoModel.from_pretrained(
            config.pretrained_config["_name_or_path"], config=config.pretrained_config
        )
        if "classifier_dropout" not in config.__dict__:
            classifier_dropout = 0.1
        else:
            classifier_dropout = (
                config.classifier_dropout
                if config.classifier_dropout is not None
                else config.hidden_dropout_prob
            )
        self.dropout = nn.Dropout(classifier_dropout)

        # Additional transformer layers
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size, nhead=config.num_attention_heads
            ),
            num_layers=2,
        )

        # For token classification, create a classifier for each task
        self.token_classifiers = nn.ModuleDict(
            {
                task: nn.Linear(config.hidden_size, num_labels)
                for task, num_labels in self.num_token_labels_dict.items()
            }
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        token_labels: Optional[dict] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        token_labels (`dict` of `torch.LongTensor` of shape `(batch_size, seq_length)`, *optional*):
            Labels for computing the token classification loss. Keys should match the tasks.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        bert_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "head_mask": head_mask,
            "inputs_embeds": inputs_embeds,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }

        if any(
            keyword in self.config.name_or_path.lower()
            for keyword in ["llama", "deberta"]
        ):
            bert_kwargs.pop("token_type_ids")
            bert_kwargs.pop("head_mask")

        outputs = self.bert(**bert_kwargs)

        # For token classification
        token_output = outputs[0]
        token_output = self.dropout(token_output)

        # Pass through additional transformer layers
        token_output = self.transformer_encoder(token_output.transpose(0, 1)).transpose(
            0, 1
        )

        # Collect the logits and compute the loss for each task
        task_logits = {}
        total_loss = 0
        for task, classifier in self.token_classifiers.items():
            logits = classifier(token_output)
            task_logits[task] = logits
            if token_labels and task in token_labels:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_token_labels_dict[task]),
                    token_labels[task].view(-1),
                )
                total_loss += loss

        if not return_dict:
            output = (task_logits,) + outputs[2:]
            return ((total_loss,) + output) if total_loss != 0 else output

        return TokenClassifierOutput(
            loss=total_loss,
            logits=task_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

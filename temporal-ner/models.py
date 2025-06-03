from transformers.modeling_outputs import TokenClassifierOutput
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.utils.data import SequentialSampler
import os
import math
import wandb
import logging
import json
from utils import set_seed, SEED, write_predictions
from seqeval.metrics import classification_report as seq_classification_report
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from transformers import get_cosine_schedule_with_warmup, BitsAndBytesConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel, AutoConfig
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class MultitaskModelForTokenClassification(PreTrainedModel):
    config_class = AutoConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, num_token_labels_dict):

        super().__init__(config)
        self.num_token_labels_dict = num_token_labels_dict
        self.config = config

        print(f"Loading model from {config.name_or_path}")
        self.model = AutoModel.from_pretrained(config.name_or_path, config=config)
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        if "classifier_dropout" not in config.__dict__:
            classifier_dropout = 0.1
        else:
            classifier_dropout = (
                config.classifier_dropout
                if config.classifier_dropout is not None
                else config.hidden_dropout_prob
            )
        self.dropout = nn.Dropout(classifier_dropout)

        # For token classification, create a classifier for each task
        self.token_classifiers = nn.ModuleDict(
            {
                task: nn.Linear(config.hidden_size, num_labels)
                for task, num_labels in num_token_labels_dict.items()
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
            labels: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            token_labels: Optional[dict] = None,
            date_indices: Optional[torch.Tensor] = None,
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
                for keyword in ["llama", "deberta", "modern"]
        ):
            bert_kwargs.pop("token_type_ids")
            bert_kwargs.pop("head_mask")

        outputs = self.model(**bert_kwargs)

        # For token classification
        token_output = outputs[0]
        token_output = self.dropout(token_output)

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


class ExtendedMultitaskModelForTokenClassification(PreTrainedModel):
    config_class = AutoConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, num_token_labels_dict):
        super().__init__(config)
        self.num_token_labels_dict = num_token_labels_dict
        self.config = config

        self.model = AutoModel.from_pretrained(config.name_or_path, config=config)
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
                for task, num_labels in num_token_labels_dict.items()
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
            date_indices: Optional[torch.Tensor] = None,
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

        outputs = self.model(**bert_kwargs)

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


class HorizonAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feed_forward1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.nonlinearity = nn.ReLU()
        self.feed_forward2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        return self.feed_forward2(self.nonlinearity(self.feed_forward1(x)))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        import torch.nn.functional as F
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


class TemporalCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)

    def forward(self, token_output, time_embedding):
        # token_output: (B, T, H), time_embedding: (B, H)
        time_as_seq = time_embedding.unsqueeze(1)  # (B, 1, H)
        attn_output, _ = self.attn(token_output, time_as_seq, time_as_seq)
        return token_output + attn_output


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union


class TemporalCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)

    def forward(self, token_output, time_embedding):
        time_seq = time_embedding.unsqueeze(1)  # (B, 1, H)
        attn_output, _ = self.attn(token_output, time_seq, time_seq)
        return token_output + attn_output


class TemporalFusion(nn.Module):
    def __init__(self, hidden_size, strategy="add", num_years=327, min_year=1700):
        super().__init__()
        self.strategy = strategy
        self.hidden_size = hidden_size
        self.min_year = min_year
        self.max_year = min_year + num_years - 1

        self.year_emb = nn.Embedding(num_years, hidden_size)

        if strategy == "concat":
            self.concat_proj = nn.Linear(hidden_size * 2, hidden_size)
        elif strategy == "film":
            self.film_gamma = nn.Linear(hidden_size, hidden_size)
            self.film_beta = nn.Linear(hidden_size, hidden_size)
        elif strategy == "adapter":
            self.adapter = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        elif strategy == "relative":
            self.relative_encoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.LayerNorm(hidden_size),
            )
            self.film_gamma = nn.Linear(hidden_size, hidden_size)
            self.film_beta = nn.Linear(hidden_size, hidden_size)
        elif strategy == "multiscale":
            self.decade_emb = nn.Embedding(1000, hidden_size)
            self.century_emb = nn.Embedding(100, hidden_size)
        elif strategy in ["early-cross-attention", "late-cross-attention"]:
            self.year_encoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU()
            )
            self.cross_attn = TemporalCrossAttention(hidden_size)

    def compute_time_embedding(self, year_index):
        if self.strategy in ["early-cross-attention", "late-cross-attention"]:
            return self.year_encoder(self.year_emb(year_index))
        elif self.strategy == "multiscale":
            year_index = year_index.long()
            year = year_index + self.min_year
            decade = (year // 10).long()
            century = (year // 100).long()
            return (
                    self.year_emb(year_index) +
                    self.decade_emb(decade) +
                    self.century_emb(century)
            )
        else:
            return self.year_emb(year_index)

    def forward(self, token_output, year_index):
        B, T, H = token_output.size()

        if self.strategy == "baseline":
            return token_output

        year_emb = self.compute_time_embedding(year_index)

        if self.strategy == "concat":
            expanded_year = year_emb.unsqueeze(1).repeat(1, T, 1)
            fused = torch.cat([token_output, expanded_year], dim=-1)
            return self.concat_proj(fused)

        elif self.strategy == "film":
            gamma = self.film_gamma(year_emb).unsqueeze(1)
            beta = self.film_beta(year_emb).unsqueeze(1)
            return gamma * token_output + beta

        elif self.strategy == "adapter":
            return token_output + self.adapter(year_emb).unsqueeze(1)

        elif self.strategy == "add":
            expanded_year = year_emb.unsqueeze(1).repeat(1, T, 1)
            return token_output + expanded_year

        elif self.strategy == "relative":
            encoded = self.relative_encoder(year_emb)
            gamma = self.film_gamma(encoded).unsqueeze(1)
            beta = self.film_beta(encoded).unsqueeze(1)
            return gamma * token_output + beta

        elif self.strategy == "multiscale":
            expanded_year = year_emb.unsqueeze(1).expand(-1, T, -1)
            return token_output + expanded_year

        elif self.strategy == "late-cross-attention":
            return self.cross_attn(token_output, year_emb)

        else:
            raise ValueError(f"Unknown fusion strategy: {self.strategy}")


class MultitaskTimeModelForTokenClassification(PreTrainedModel):
    config_class = AutoConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, num_token_labels_dict, temporal_fusion_strategy="add", num_years=327):
        super().__init__(config)
        self.config = config
        self.num_token_labels_dict = num_token_labels_dict
        self.temporal_fusion_strategy = temporal_fusion_strategy
        self.model = AutoModel.from_pretrained(config.name_or_path, config=config)
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        self.num_years = num_years

        classifier_dropout = getattr(config, "classifier_dropout", 0.1) or config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)

        self.temporal_fusion = TemporalFusion(config.hidden_size, strategy=self.temporal_fusion_strategy,
                                              num_years=num_years)

        self.token_classifiers = nn.ModuleDict({
            task: nn.Linear(config.hidden_size, num_labels)
            for task, num_labels in num_token_labels_dict.items()
        })

        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            token_labels: Optional[dict] = None,
            date_indices: Optional[torch.Tensor] = None,
            year_index: Optional[torch.Tensor] = None,
            decade_index: Optional[torch.Tensor] = None,
            century_index: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embeddings(input_ids)

        # Early cross-attention fusion
        if self.temporal_fusion_strategy == "early-cross-attention":
            year_emb = self.temporal_fusion.compute_time_embedding(year_index)  # (B, H)
            inputs_embeds = self.temporal_fusion.cross_attn(inputs_embeds, year_emb)

        bert_kwargs = {
            "inputs_embeds": inputs_embeds if self.temporal_fusion_strategy == "early-cross-attention" else None,
            "input_ids": input_ids if self.temporal_fusion_strategy != "early-cross-attention" else None,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "head_mask": head_mask,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }

        if any(keyword in self.config.name_or_path.lower() for keyword in ["llama", "deberta"]):
            bert_kwargs.pop("token_type_ids", None)
            bert_kwargs.pop("head_mask", None)

        outputs = self.model(**bert_kwargs)
        token_output = self.dropout(outputs[0])  # (B, T, H)
        hidden_states = list(outputs.hidden_states) if output_hidden_states else None

        # Apply fusion after transformer if needed
        if self.temporal_fusion_strategy not in ["baseline", "early-cross-attention"]:
            token_output = self.temporal_fusion(token_output, year_index)
            if output_hidden_states:
                hidden_states.append(token_output)  # add the final fused state

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
            hidden_states=tuple(hidden_states) if hidden_states is not None else None,
            attentions=outputs.attentions if output_attentions else None,
        )


class MultitaskExtendedDateSwigluModelForTokenClassification(PreTrainedModel):
    config_class = AutoConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, num_token_labels_dict):
        super().__init__(config)
        self.num_token_labels_dict = num_token_labels_dict
        self.config = config

        self.model = AutoModel.from_pretrained(config.name_or_path, config=config)
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        if "classifier_dropout" not in config.__dict__:
            classifier_dropout = 0.1
        else:
            classifier_dropout = (
                config.classifier_dropout
                if config.classifier_dropout is not None
                else config.hidden_dropout_prob
            )
        self.dropout = nn.Dropout(classifier_dropout)

        self.date_embedding = nn.Embedding(
            512, config.hidden_size
        )  # Assuming a max of 512 unique dates

        # Horizon Adapter for date embeddings
        self.horizon_adapter = HorizonAdapter(config)

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
                for task, num_labels in num_token_labels_dict.items()
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
            labels: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            token_labels: Optional[dict] = None,
            date_indices: Optional[torch.Tensor] = None,
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

        outputs = self.model(**bert_kwargs)

        # For token classification
        token_output = outputs[0]
        token_output = self.dropout(token_output)

        # Pass through additional transformer layers
        token_output = self.transformer_encoder(token_output.transpose(0, 1)).transpose(
            0, 1
        )

        # Embed date_indices and adapt using Horizon Adapter
        if date_indices is not None:
            date_embeddings = self.date_embedding(date_indices)
            date_embeddings = self.horizon_adapter(date_embeddings)
            token_output = (
                    token_output + date_embeddings
            )  # Combining embeddings, you can use other methods as well

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


class MultitaskExtendedSeparatedDateSwigluModelForTokenClassification(PreTrainedModel):
    config_class = AutoConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, num_token_labels_dict):
        super().__init__(config)
        self.num_token_labels_dict = num_token_labels_dict
        self.config = config

        self.model = AutoModel.from_pretrained(config.name_or_path, config=config)
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        if "classifier_dropout" not in config.__dict__:
            classifier_dropout = 0.1
        else:
            classifier_dropout = (
                config.classifier_dropout
                if config.classifier_dropout is not None
                else config.hidden_dropout_prob
            )
        self.dropout = nn.Dropout(classifier_dropout)

        # Embedding layers for date components
        self.year_embedding = nn.Embedding(
            3000, config.hidden_size
        )  # Assuming years from 0 to 2999
        self.month_embedding = nn.Embedding(12, config.hidden_size)
        self.day_embedding = nn.Embedding(31, config.hidden_size)

        # Horizon Adapter for date embeddings
        self.horizon_adapter = HorizonAdapter(config)

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
                for task, num_labels in num_token_labels_dict.items()
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
            labels: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            token_labels: Optional[dict] = None,
            date_indices: Optional[torch.Tensor] = None,
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

        outputs = self.model(**bert_kwargs)

        # For token classification
        token_output = outputs[0]
        token_output = self.dropout(token_output)

        # Pass through additional transformer layers
        token_output = self.transformer_encoder(token_output.transpose(0, 1)).transpose(
            0, 1
        )

        # Embed date_indices and adapt using Horizon Adapter
        if date_indices is not None:
            year_embeddings = self.year_embedding(date_indices[:, 0])
            month_embeddings = self.month_embedding(date_indices[:, 1])
            day_embeddings = self.day_embedding(date_indices[:, 2])

            date_embeddings = year_embeddings + month_embeddings + day_embeddings
            date_embeddings = self.horizon_adapter(date_embeddings)

            token_output = token_output + date_embeddings.unsqueeze(
                1
            )  # Combining embeddings, you can use other methods as well

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


class DatePositionEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, d_model)

    def forward(self, date_indices):
        # date_indices assumed to be [batch_size, 3] where columns are year, month, day
        position_indices = (
                date_indices[:, 0] * 372 + date_indices[:, 1] * 31 + date_indices[:, 2]
        )
        return self.position_embedding(position_indices)


def calculate_relative_positions(date_indices):
    """
    Calculate the relative positions of dates in terms of days.

    Args:
        date_indices (torch.Tensor): A tensor of shape [batch_size, seq_len, 3] where each entry contains [year, month, day].

    Returns:
        torch.Tensor: A tensor of shape [batch_size, seq_len, seq_len] containing relative date distances in days.
    """
    batch_size, seq_len, _ = date_indices.size()

    # Initialize a tensor to store relative positions
    relative_positions = torch.zeros(
        (batch_size, seq_len, seq_len), dtype=torch.float32
    )

    # Iterate over each batch
    for i in range(batch_size):
        for j in range(seq_len):
            for k in range(seq_len):
                if j == k:
                    relative_positions[i, j, k] = 0
                else:
                    date1 = datetime(
                        year=date_indices[i, j, 0].item(),
                        month=date_indices[i, j, 1].item() + 1,
                        day=date_indices[i, j, 2].item() + 1,
                    )
                    date2 = datetime(
                        year=date_indices[i, k, 0].item(),
                        month=date_indices[i, k, 1].item() + 1,
                        day=date_indices[i, k, 2].item() + 1,
                    )
                    diff = (date1 - date2).days
                    relative_positions[i, j, k] = diff

    # Optional: Normalize the relative positions
    max_value = torch.max(torch.abs(relative_positions))
    if max_value > 0:
        relative_positions = relative_positions / max_value

    return relative_positions


class RelativeDatePositionEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.relative_positions = nn.Parameter(torch.zeros(max_len, d_model))

    def forward(self, date_indices):
        batch_size = date_indices.size(0)
        # Compute relative positions
        # Here we'll need a more complex logic to calculate pairwise date distances
        # Example only, assumes some function `calculate_relative_positions` exists
        relative_positions = calculate_relative_positions(
            date_indices
        )  # [batch_size, seq_len, seq_len]
        return torch.matmul(relative_positions, self.relative_positions)


class MultitaskPositionExtendedDateSwigluModelForTokenClassification(PreTrainedModel):
    config_class = AutoConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, num_token_labels_dict):
        super().__init__(config)
        self.num_token_labels_dict = num_token_labels_dict
        self.config = config

        self.model = AutoModel.from_pretrained(config.name_or_path, config=config)
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        if "classifier_dropout" not in config.__dict__:
            classifier_dropout = 0.1
        else:
            classifier_dropout = (
                config.classifier_dropout
                if config.classifier_dropout is not None
                else config.hidden_dropout_prob
            )
        self.dropout = nn.Dropout(classifier_dropout)

        self.date_embedding = nn.Embedding(
            512, config.hidden_size
        )  # Assuming a max of 512 unique dates

        # Horizon Adapter for date embeddings
        self.horizon_adapter = HorizonAdapter(config)

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
                for task, num_labels in num_token_labels_dict.items()
            }
        )

        # Position Encoding for dates
        self.date_position_encoding = DatePositionEncoding(
            max_len=3000 * 12 * 31, d_model=config.hidden_size
        )
        # Or use RelativeDatePositionEncoding
        # self.relative_date_position_encoding = RelativeDatePositionEncoding(max_len=365, d_model=config.hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            token_labels: Optional[dict] = None,
            date_indices: Optional[torch.Tensor] = None,
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

        outputs = self.model(**bert_kwargs)

        # For token classification
        token_output = outputs[0]
        token_output = self.dropout(token_output)

        # Pass through additional transformer layers
        token_output = self.transformer_encoder(token_output.transpose(0, 1)).transpose(
            0, 1
        )

        # Embed date_indices and adapt using Horizon Adapter
        if date_indices is not None:
            date_embeddings = self.date_embedding(date_indices)
            date_embeddings = self.horizon_adapter(date_embeddings)

            # Combine position encoding
            date_position_encoded = self.date_position_encoding(date_indices)
            # Or use relative position encoding
            # date_position_encoded = self.relative_date_position_encoding(date_indices)
            date_embeddings = date_embeddings + date_position_encoded

            token_output = token_output + date_embeddings.unsqueeze(
                1
            )  # Combining embeddings, you can use other methods as well

        # Collect the logits and compute the loss for each task
        task_logits = {}
        total_loss = 0
        for task, classifier in self.token_classifiers.items():
            logits = classifier(token_output)
            task_logits[task] = logits
            if token_labels and task in token_labels:
                loss_fct = nn.CrossEntropyLoss()
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


class MultitaskPositionExtendedSeparatedDateSwigluModelForTokenClassification(
    PreTrainedModel
):
    config_class = AutoConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, num_token_labels_dict):
        super().__init__(config)
        self.num_token_labels_dict = num_token_labels_dict
        self.config = config

        self.model = AutoModel.from_pretrained(config.name_or_path, config=config)
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        if "classifier_dropout" not in config.__dict__:
            classifier_dropout = 0.1
        else:
            classifier_dropout = (
                config.classifier_dropout
                if config.classifier_dropout is not None
                else config.hidden_dropout_prob
            )
        self.dropout = nn.Dropout(classifier_dropout)

        # Embedding layers for date components
        self.year_embedding = nn.Embedding(
            3000, config.hidden_size
        )  # Assuming years from 0 to 2999
        self.month_embedding = nn.Embedding(12, config.hidden_size)
        self.day_embedding = nn.Embedding(31, config.hidden_size)

        # Horizon Adapter for date embeddings
        self.horizon_adapter = HorizonAdapter(config)

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
                for task, num_labels in num_token_labels_dict.items()
            }
        )

        # Position Encoding for dates
        self.date_position_encoding = DatePositionEncoding(
            max_len=3000 * 12 * 31, d_model=config.hidden_size
        )
        # Or use RelativeDatePositionEncoding
        # self.relative_date_position_encoding = RelativeDatePositionEncoding(max_len=365, d_model=config.hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            token_labels: Optional[dict] = None,
            date_indices: Optional[torch.Tensor] = None,
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

        outputs = self.model(**bert_kwargs)

        # For token classification
        token_output = outputs[0]
        token_output = self.dropout(token_output)

        # Pass through additional transformer layers
        token_output = self.transformer_encoder(token_output.transpose(0, 1)).transpose(
            0, 1
        )

        # Embed date_indices and adapt using Horizon Adapter
        if date_indices is not None:
            year_embeddings = self.year_embedding(date_indices[:, 0])
            month_embeddings = self.month_embedding(date_indices[:, 1])
            day_embeddings = self.day_embedding(date_indices[:, 2])

            date_embeddings = year_embeddings + month_embeddings + day_embeddings
            date_embeddings = self.horizon_adapter(date_embeddings)

            # Combine position encoding
            date_position_encoded = self.date_position_encoding(date_indices)
            # Or use relative position encoding
            # date_position_encoded = self.relative_date_position_encoding(date_indices)
            date_embeddings = date_embeddings + date_position_encoded

            token_output = token_output + date_embeddings.unsqueeze(
                1
            )  # Combining embeddings, you can use other methods as well

        # Collect the logits and compute the loss for each task
        task_logits = {}
        total_loss = 0
        for task, classifier in self.token_classifiers.items():
            logits = classifier(token_output)
            task_logits[task] = logits
            if token_labels and task in token_labels:
                loss_fct = nn.CrossEntropyLoss()
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


def _move_model_to_device(model, device):
    """Move a model to a device (CPU or GPU)."""
    model = model.to(device)
    # Moving a model to an XLA device disconnects the tied weights, so we have to retie them.
    # model.tie_weights()
    return model


def train(
        args,
        train_dataset,
        dev_dataset,
        test_dataset,
        model,
        tokenizer,
        optimizer,
        label_map,
        model_class,
):
    if args.wandb:
        wandb.watch(
            model, log="all", log_freq=10
        )  # Added log_freq to ensure frequent logging

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    best_score = 0.0
    best_output_dir = None

    args.train_batch_size = args.train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )

    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    t_total = (
            math.ceil(len(train_dataset) / args.train_batch_size) * args.epochs
    )  # assume 10 epochs
    # 10% of training steps for warmup
    num_warmup_steps = math.ceil(t_total * 0.1)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(
            os.path.join(args.model_name_or_path, "optimizer.pt")
    ) and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")), )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (
                len(train_dataloader) // args.gradient_accumulation_steps
        )
        steps_trained_in_current_epoch = global_step % (
                len(train_dataloader) // args.gradient_accumulation_steps
        )

        logger.info(
            "  Continuing training from checkpoint, will skip to saved global_step"
        )
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info(
            "  Will skip the first %d steps in the first epoch",
            steps_trained_in_current_epoch,
        )

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    set_seed(SEED)  # Added here for reproductibility

    results_devset = {result: [] for result in ["global", "token-level"]}
    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            model = _move_model_to_device(model, args.device)
            inputs = {
                "input_ids": batch["input_ids"].to(args.device),
                "attention_mask": batch["attention_mask"].to(args.device),
                "token_labels": {
                    task: labels.to(args.device)
                    for task, labels in batch["token_targets"].items()
                },
                "date_indices": batch["date_indices"].to(args.device) if "date_indices" in batch else None,
                "year_index": batch["year_index"].to(args.device) if "year_index" in batch else None,
                "decade_index": batch["decade_index"].to(args.device) if "decade_index" in batch else None,
                "century_index": batch["century_index"].to(args.device) if "century_index" in batch else None,

            }

            # Convert tensors to float16 if using fp16
            if args.fp16:
                for key in inputs:
                    if isinstance(inputs[key], dict):
                        for subkey in inputs[key]:
                            inputs[key][subkey] = inputs[key][subkey].to(torch.float16)
                    else:
                        inputs[key] = inputs[key].to(torch.float16)

            if "bert" in args.model_name_or_path.lower() and not "modern" in args.model_name_or_path.lower():
                inputs["token_type_ids"] = batch["token_type_ids"].to(args.device)
            elif "xlnet" in args.model_name_or_path.lower():
                inputs["token_type_ids"] = batch["token_type_ids"].to(args.device)
            else:
                if "token_type_ids" in inputs:
                    del inputs["token_type_ids"]
            # if "token_type_ids" in inputs:
            #     print(f"token_type_ids: {inputs['token_type_ids'].shape}")
            # else:
            #     print("token_type_ids not found in inputs")

            outputs = model(**inputs)
            loss = outputs.loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (
                        args.local_rank in [-1, 0]
                        and args.logging_steps > 0
                        and global_step % args.logging_steps == 0
                ):
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results, words_list, preds_list, report_bin, report_class = evaluate(
                            args, model, dev_dataset, label_map, prefix="dev", tokenizer=tokenizer
                        )

                        write_predictions(
                            args,
                            args.output_dir,
                            dev_dataset.get_filename(),
                            words_list,
                            preds_list,
                        )

                        # Log to wandb
                        if args.wandb:
                            for key, value in results.items():
                                wandb.log({f"eval_{key}": value})

                        results_devset["global"].append(results)
                        results_devset["token-level"].append(report_class)

                        # Save best model based on F1
                        # {'loss': 0.17091961527404811, 'NE-COARSE-LIT_precision': 0.6052540615278258,
                        #  'NE-COARSE-LIT_recall': 0.7160089961153139, 'NE-COARSE-LIT_f1': 0.6559895101620306,
                        #  'NE-FINE-COMP_precision': 0.5385450597176982, 'NE-FINE-COMP_recall': 0.6836664369400414,
                        #  'NE-FINE-COMP_f1': 0.6024901305800183}
                        current_f1 = results.get("NE-COARSE-LIT_f1")
                        logger.info(f"Current F1: {current_f1}")

                        if current_f1 > best_score:
                            best_score = current_f1
                            best_output_dir = os.path.join(args.output_dir, "best_checkpoint")
                            if not os.path.exists(best_output_dir):
                                os.makedirs(best_output_dir)
                            model_to_save = model.module if hasattr(model, "module") else model
                            model_to_save.save_pretrained(best_output_dir, safe_serialization=False)
                            tokenizer.save_pretrained(best_output_dir)
                            torch.save(args, os.path.join(best_output_dir, "training_args.bin"))
                            logger.info("Saved new best model checkpoint to %s", best_output_dir)

                            # Save model checkpoint
                            output_dir = os.path.join(
                                args.output_dir, "checkpoint-{}".format(global_step)
                            )
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)

                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving model checkpoint to %s", output_dir)

                            torch.save(
                                optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                            )
                            torch.save(
                                scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                            )
                            logger.info(
                                "Saving optimizer and scheduler states to %s", output_dir
                            )

                    if args.wandb:
                        wandb.log(
                            {
                                "lr": scheduler.get_last_lr()[0],
                                "loss": (tr_loss - logging_loss) / args.logging_steps,
                            }
                        )
                    logging_loss = tr_loss

                if (
                        args.local_rank in [-1, 0]
                        and args.save_steps > 0
                        and global_step % args.save_steps == 0
                ):
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-{}".format(global_step)
                    )
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(
                        optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                    )
                    torch.save(
                        scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                    )
                    logger.info(
                        "Saving optimizer and scheduler states to %s", output_dir
                    )

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    with open(os.path.join(args.output_dir, "best_checkpoint_path.txt"), "w") as f:
        f.write(best_output_dir)

    print(f"Best model saved to {best_output_dir} - loading..")
    num_sequence_labels, num_token_labels_dict, num_years = train_dataset.get_info()

    model_class_obj = globals()[model_class]
    config = AutoConfig.from_pretrained(best_output_dir, problem_type="single_label_classification",
                                        local_files_only=True, )
    # Use the class object to instantiate the model
    model = model_class_obj(config, num_token_labels_dict=num_token_labels_dict,
                            temporal_fusion_strategy=args.temporal_fusion_strategy, num_years=num_years)
    #
    # model = MultitaskTimeModelForTokenClassification(config, num_token_labels_dict,
    #                                                  temporal_fusion_strategy=args.temporal_fusion_strategy,
    #                                                  num_years=num_years)

    checkpoint = torch.load(os.path.join(best_output_dir, "pytorch_model.bin"))
    print(f"Loading {checkpoint.keys()}...")
    model.load_state_dict(checkpoint)

    # model = MultitaskModelForTokenClassification.from_pretrained(
    #     best_output_dir,
    #     num_token_labels_dict=num_token_labels_dict,
    # )
    model = model.to(args.device)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    results, words_list, preds_list, report_bin, report_class = evaluate(
        args, model, test_dataset, label_map, prefix="test", tokenizer=tokenizer
    )

    write_predictions(
        args, args.output_dir, test_dataset.get_filename(), words_list, preds_list
    )

    results_testset = {"global": results, "token-level": report_class}

    all_results = {"dev": results_devset, "test": results_testset}
    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f)
    return global_step, tr_loss / global_step


def evaluate(args, model, dataset, label_map, prefix="", tokenizer=None):
    args.eval_batch_size = args.eval_batch_size * max(1, args.n_gpu)
    eval_sampler = (
        SequentialSampler(dataset)
        if args.local_rank == -1
        else DistributedSampler(dataset)
    )

    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    out_token_ids, out_token_preds = {}, {}
    sentences, text_sentences = None, None

    model.eval()
    all_dates = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            inputs = {
                "input_ids": batch["input_ids"].to(args.device),
                "attention_mask": batch["attention_mask"].to(args.device),
                "token_labels": {
                    task: labels.to(args.device)
                    for task, labels in batch["token_targets"].items()
                },
                "date_indices": batch["date_indices"].to(args.device) if "date_indices" in batch else None,
                "year_index": batch["year_index"].to(args.device) if "year_index" in batch else None,
                "decade_index": batch["decade_index"].to(args.device) if "decade_index" in batch else None,
                "century_index": batch["century_index"].to(args.device) if "century_index" in batch else None,
            }
            # Determine model type and handle token_type_ids if necessary
            if isinstance(model, torch.nn.DataParallel):
                actual_model = model.module
            else:
                actual_model = model

            model_name = actual_model.config._name_or_path.lower()

            if "bert" in args.model_name_or_path.lower() and not "modern" in args.model_name_or_path.lower():
                inputs["token_type_ids"] = batch["token_type_ids"].to(args.device)
            elif "xlnet" in args.model_name_or_path.lower():
                inputs["token_type_ids"] = batch["token_type_ids"].to(args.device)

            if "date" in model_name:
                inputs["date_indices"] = batch["date_indices"].to(args.device)

            if "bert" in model_name or "xlnet" in model_name:
                inputs["token_type_ids"] = batch["token_type_ids"].to(args.device)

            if args.fp16:
                for key in inputs:
                    if isinstance(inputs[key], dict):
                        for subkey in inputs[key]:
                            inputs[key][subkey] = inputs[key][subkey].to(torch.float16)
                    else:
                        inputs[key] = inputs[key].to(torch.float16)

            outputs = model(**inputs)

            tmp_eval_loss = outputs.loss

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1

        in_date_indices = inputs["date_indices"].to(torch.float32).detach().cpu().numpy()
        in_dates = [tokenizer.convert_ids_to_tokens(ids) for ids in in_date_indices]
        dates = [text.split(" ") for text in batch["date"]]
        all_dates.extend(dates)

        for task, logits in outputs.logits.items():
            if task not in out_token_preds:
                out_token_preds[task] = logits.to(torch.float32).detach().cpu().numpy()
                out_token_ids[task] = (
                    inputs["token_labels"][task]
                    .to(torch.float32)
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                out_token_preds[task] = np.append(
                    out_token_preds[task],
                    logits.to(torch.float32).detach().cpu().numpy(),
                    axis=0,
                )
                out_token_ids[task] = np.append(
                    out_token_ids[task],
                    inputs["token_labels"][task]
                    .to(torch.float32)
                    .detach()
                    .cpu()
                    .numpy(),
                    axis=0,
                )

        if sentences is None:
            out_input_ids = inputs["input_ids"].to(torch.float32).detach().cpu().numpy()
            sentences = [
                tokenizer.convert_ids_to_tokens(input_ids)
                for input_ids in out_input_ids
            ]
            text_sentences = [text.split(" ") for text in batch["sequence"]]
        else:
            sentences = np.append(
                sentences,
                [
                    tokenizer.convert_ids_to_tokens(input_ids)
                    for input_ids in inputs["input_ids"]
                .to(torch.float32)
                .detach()
                .cpu()
                .numpy()
                ],
                axis=0,
            )
            text_sentences = text_sentences + [
                text.split(" ") for text in batch["sequence"]
            ]

    eval_loss = eval_loss / nb_eval_steps

    results = {"loss": eval_loss}
    all_out_label_list = {}
    all_preds_list = {}
    all_words_list = {}

    for task in out_token_preds:
        out_token_preds[task] = np.argmax(out_token_preds[task], axis=2)
        label_map_task = {i: label for label, i in label_map[task].items()}

        out_label_list = [[] for _ in range(out_token_ids[task].shape[0])]
        preds_list = [[] for _ in range(out_token_ids[task].shape[0])]
        words_list = [[] for _ in range(out_token_ids[task].shape[0])]

        for idx_sentence, item in enumerate(
                zip(text_sentences, out_token_ids[task], out_token_preds[task])
        ):
            text_sentence, out_label_ids, out_label_preds = item
            word_ids = tokenizer(text_sentence, is_split_into_words=True).word_ids()
            for idx, word in enumerate(text_sentence):
                beginning_index = word_ids.index(idx)
                try:
                    out_label_list[idx_sentence].append(
                        label_map_task[out_label_ids[beginning_index]]
                    )
                except BaseException:
                    out_label_list[idx_sentence].append("O")
                try:
                    preds_list[idx_sentence].append(
                        label_map_task[out_label_preds[beginning_index]]
                    )
                except BaseException:
                    preds_list[idx_sentence].append("O")
                words_list[idx_sentence].append(word)

        all_out_label_list[task] = out_label_list
        all_preds_list[task] = preds_list
        all_words_list[task] = words_list

        results[f"{task}_precision"] = precision_score(out_label_list, preds_list)
        results[f"{task}_precision"] = precision_score(out_label_list, preds_list)
        results[f"{task}_recall"] = recall_score(out_label_list, preds_list)
        results[f"{task}_f1"] = f1_score(out_label_list, preds_list)

    logger.info("Evaluation for named entity recognition & classification.")

    report_class = {
        task: seq_classification_report(all_out_label_list[task], all_preds_list[task], digits=4)
        for task in all_out_label_list
    }
    for task in report_class:
        logger.info("\n%s", report_class[task])

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
        if args.wandb:
            wandb.log({f"eval_{key}": results[key]})

    from collections import defaultdict

    by_year = defaultdict(lambda: {"labels": [], "preds": []})
    by_decade = defaultdict(lambda: {"labels": [], "preds": []})

    # Group labels and predictions by year and decade
    for i, date_parts in enumerate(all_dates):
        year = date_parts[0]
        decade = year[:3] + "0s"

        for task in all_preds_list:
            by_year[year]["labels"].append(all_out_label_list[task][i])
            by_year[year]["preds"].append(all_preds_list[task][i])
            by_decade[decade]["labels"].append(all_out_label_list[task][i])
            by_decade[decade]["preds"].append(all_preds_list[task][i])

    rows = []

    # Process by year
    for year in sorted(by_year):
        report = seq_classification_report(by_year[year]["labels"], by_year[year]["preds"], output_dict=True)
        for label, metrics in report.items():
            if label in {"macro avg", "weighted avg"}:
                continue
            rows.append({
                "scope": "year",
                "time": year,
                "entity_type": label,
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1-score"], 4)
            })

    # Process by decade
    for decade in sorted(by_decade):
        report = seq_classification_report(by_decade[decade]["labels"], by_decade[decade]["preds"], output_dict=True)
        for label, metrics in report.items():
            if label in {"macro avg", "weighted avg"}:
                continue
            rows.append({
                "scope": "decade",
                "time": decade,
                "entity_type": label,
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1-score"], 4)
            })

    # Save to TSV
    df = pd.DataFrame(rows)
    print(df)
    output_path = os.path.join(args.output_dir,
                               args.output_dir.split('/')[-1] + f"_{prefix}_ner_temporal_entity_scores.tsv")
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved per-year and per-decade evaluation to {output_path}")

    # Report per year
    # logger.info("***** Classification Report by Year *****")
    # for year in sorted(by_year):
    #     try:
    #         report = seq_classification_report(
    #             by_year[year]["labels"], by_year[year]["preds"], digits=4
    #         )
    #         logger.info(f"\nYear {year}:\n{report}")
    #     except Exception as e:
    #         logger.warning(f"Could not compute report for {year}: {e}")
    #
    # # Report per decade
    # logger.info("***** Classification Report by Decade *****")
    # for decade in sorted(by_decade):
    #     try:
    #         report = seq_classification_report(
    #             by_decade[decade]["labels"], by_decade[decade]["preds"], digits=4
    #         )
    #         logger.info(f"\nDecade {decade}:\n{report}")
    #     except Exception as e:
    #         logger.warning(f"Could not compute report for {decade}: {e}")

    return results, all_words_list, all_preds_list, None, report_class

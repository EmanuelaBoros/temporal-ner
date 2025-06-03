from utils import write_predictions
import logging
from typing import Optional, Tuple, Union
import json
import numpy as np
from seqeval.metrics import classification_report as seq_classification_report
from tqdm import tqdm, trange
import os
import logging
import math
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from seqeval.metrics import f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from transformers import (
    BertModel,
    BertTokenizer,
    BertPreTrainedModel,
    AutoModel,
    PreTrainedModel,
    AutoModelForPreTraining,
)
from transformers.modeling_outputs import TokenClassifierOutput

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from transformer import TransformerEncoder, MultiHeadAttn, TransformerLayer
from utils import set_seed, SEED
import wandb
from heinsen_routing.heinsen_routing import EfficientVectorRouting

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch.nn.functional as F


def _move_model_to_device(model, device):
    """Move a model to a device (CPU or GPU)."""
    model = model.to(device)
    # Moving a model to an XLA device disconnects the tied weights, so we have to retie them.
    # model.tie_weights()
    return model


class HorizonAdapter(nn.Module):
    def __init__(self, input_dim, adapter_dim):
        super(HorizonAdapter, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.Dropout(0.5),  # Added dropout layer
        )

    def forward(self, x):
        return self.adapter(x)


class CapsuleLayer(nn.Module):
    def __init__(self, n_out, d_inp, d_out, n_iters=3):
        super(CapsuleLayer, self).__init__()
        self.n_out = n_out
        self.d_inp = d_inp
        self.d_out = d_out
        self.n_iters = n_iters
        self.layer_norm = nn.LayerNorm(n_out * d_out)
        self.routing = EfficientVectorRouting(
            n_inp=d_inp, n_out=n_out, d_inp=d_inp, d_out=d_out, n_iters=n_iters
        )

    def forward(self, x):
        # x should have the shape [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = x.size()

        # Ensure the hidden_dim matches the expected input dimension
        if hidden_dim != self.d_inp:
            raise ValueError(
                f"Input hidden dimension {hidden_dim} does not match expected input dimension {self.d_inp}."
            )

        device = x.device

        # Flatten the input to [batch_size * seq_len, d_inp]
        x_reshaped = x.view(-1, hidden_dim).to(device)  # [3072, 768]

        # Apply routing
        v_j = self.routing(x_reshaped)  # [512, 32]

        # Ensure the output shape is [batch_size * seq_len, n_out, d_out]
        expected_output_shape = (self.n_out, self.d_out)
        if v_j.shape != expected_output_shape:
            raise ValueError(
                f"Expected output shape {expected_output_shape}, but got {v_j.shape}"
            )

        # Reshape back to [batch_size, seq_len, n_out * d_out]
        v_j = v_j.view(batch_size, seq_len, -1)  # [6, 512, 512 * 32]

        # Apply layer normalization
        v_j = self.layer_norm(v_j)

        return v_j

    @staticmethod
    def squash(x, epsilon=1e-7):
        squared_norm = (x ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * x / torch.sqrt(squared_norm + epsilon)


class TextDecoder(nn.Module):
    def __init__(self, num_capsules, capsule_dim, output_dim):
        super(TextDecoder, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.fc1 = nn.Linear(num_capsules * capsule_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, output_dim)
        self.layer_norm1 = nn.LayerNorm(512)  # Added Layer Normalization
        self.layer_norm2 = nn.LayerNorm(1024)  # Added Layer Normalization

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the capsule output
        x = F.relu(self.fc1(x))
        x = self.layer_norm1(x)  # Apply Layer Normalization
        x = F.relu(self.fc2(x))
        x = self.layer_norm2(x)  # Apply Layer Normalization
        x = self.fc3(x)  # Output a vector matching the input embedding size
        return x


class BertForTokenClassificationWithAdapters(BertPreTrainedModel):
    def __init__(self, config, capsule_dim=16, routing_iters=3):
        super().__init__(config)
        self.num_classes = config.num_labels
        self.capsule_dim = capsule_dim
        self.routing_iters = routing_iters

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Transformation matrix for capsules
        self.W = nn.Parameter(torch.randn(config.hidden_size, capsule_dim))

        # Class capsules
        self.class_capsules = nn.Parameter(torch.randn(self.num_classes, capsule_dim))

        self.init_weights()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        # Transform token embeddings to capsule space
        U = torch.matmul(sequence_output, self.W)

        # Dynamic routing
        logits = self.dynamic_routing(U)

        # Reshape V for classification
        # logits = V.view(-1, self.num_classes)

        loss = None
        if labels is not None:
            # Flatten labels to match logits shape
            labels = labels.view(-1)
            # Create a mask to ignore the special tokens (`-100`)
            # active_loss = labels != -100
            # active_logits = logits[active_loss]
            # active_labels = labels[active_loss]
            loss_fct = nn.CrossEntropyLoss()
            # loss = loss_fct(active_logits, active_labels)
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def dynamic_routing(self, U):
        batch_size = U.size(0)
        num_tokens = U.size(1)

        # Initialize routing logits to zero
        B = torch.zeros(batch_size, num_tokens, self.num_classes, device=U.device)

        for _ in range(self.routing_iters):
            # Routing coefficients (softmax over the class capsules)
            C = torch.softmax(B, dim=2)

            # Weighted sum of input capsules
            S = (C.unsqueeze(3) * U.unsqueeze(2)).sum(dim=1)

            # Squash non-linearity
            V = self.squash(S)

            # Update routing logits
            B = B + torch.matmul(U, V.transpose(1, 2))

        return V

    def squash(self, S):
        norm = torch.norm(S, dim=2, keepdim=True)
        scale = norm ** 2 / (1 + norm ** 2)
        V = scale * S / (norm + 1e-8)
        return V


class HorizonAdapter(nn.Module):
    def __init__(self, hidden_size):
        super(HorizonAdapter, self).__init__()
        self.feed_forward1 = nn.Linear(hidden_size, hidden_size)
        self.nonlinearity = nn.ReLU()
        self.feed_forward2 = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.nonlinearity(self.feed_forward1(x))
        out = self.feed_forward2(out)
        out = self.layer_norm(out + x)
        return self.dropout(out)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return self.relu(out)


class CapsuleNetwork(nn.Module):
    def __init__(self, hidden_size):
        super(CapsuleNetwork, self).__init__()
        self.conv_features = DepthwiseSeparableConv(
            hidden_size, hidden_size, kernel_size=3, padding=1
        )
        self.primary_caps = nn.Linear(hidden_size, hidden_size)
        self.time_caps = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        conv_output = self.conv_features(x.transpose(1, 2))
        primary_caps_output = self.primary_caps(conv_output.transpose(1, 2))
        time_caps_output = self.time_caps(primary_caps_output)
        time_caps_output = self.layer_norm(time_caps_output + x)
        return self.dropout(time_caps_output)


class MultiCapsuleAttention(nn.Module):
    def __init__(self, hidden_size):
        super(MultiCapsuleAttention, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, time_caps_output):
        attn_output, _ = self.multi_head_attention(
            x, time_caps_output, time_caps_output
        )
        attn_output = self.layer_norm(attn_output + x)
        return self.dropout(attn_output)


class LongHorizonBertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(LongHorizonBertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.horizon_adapter = HorizonAdapter(config.hidden_size)
        self.capsule_network = CapsuleNetwork(config.hidden_size)
        self.multi_capsule_attention = MultiCapsuleAttention(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
            self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        sequence_output = self.horizon_adapter(sequence_output)
        time_caps_output = self.capsule_network(sequence_output)
        sequence_output = self.multi_capsule_attention(
            sequence_output, time_caps_output
        )
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class StackedBertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config, num_layers, d_model, n_head, feedforward_dim, dropout,
                 after_norm=True, attn_type='adatrans', bi_embed=None,
                 fc_dropout=0.3, pos_embed=None, scale=False, dropout_attn=None):
        super(StackedBertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        embed_size = 512
        self.in_fc = nn.Linear(embed_size, d_model)
        self.transformer = TransformerEncoder(num_layers, d_model, n_head, feedforward_dim, dropout,
                                              after_norm=after_norm, attn_type=attn_type,
                                              scale=scale, dropout_attn=dropout_attn,
                                              pos_embed=pos_embed)

        self.self_attn = MultiHeadAttn(d_model, n_head)

        self.pooling_methods = ['max', 'mean', 'max-mean']

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
            self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        mask = sequence_output.ne(0)

        chars = self.in_fc(sequence_output)
        chars = self.transformer(chars, mask)
        sequence_output = self.fc_dropout(chars)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class HorizonAdapter(nn.Module):
    def __init__(self, hidden_size):
        super(HorizonAdapter, self).__init__()
        self.feed_forward1 = nn.Linear(hidden_size, hidden_size)
        self.nonlinearity = nn.GELU()  # GELU is often more effective than ReLU
        self.feed_forward2 = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.nonlinearity(self.feed_forward1(x))
        out = self.feed_forward2(out)
        out = self.layer_norm(out + x)
        return self.dropout(out)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return self.act(out)


class CapsuleNetwork(nn.Module):
    def __init__(self, hidden_size):
        super(CapsuleNetwork, self).__init__()
        self.conv_features = DepthwiseSeparableConv(
            hidden_size, hidden_size, kernel_size=3, padding=1
        )
        self.primary_caps = nn.Linear(hidden_size, hidden_size)
        self.time_caps = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        conv_output = self.conv_features(x.transpose(1, 2))
        primary_caps_output = self.primary_caps(conv_output.transpose(1, 2))
        time_caps_output = self.time_caps(primary_caps_output)
        time_caps_output = self.layer_norm(time_caps_output + x)
        return self.dropout(time_caps_output)


class MultiCapsuleAttention(nn.Module):
    def __init__(self, hidden_size):
        super(MultiCapsuleAttention, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, time_caps_output):
        attn_output, _ = self.multi_head_attention(
            x, time_caps_output, time_caps_output
        )
        attn_output = self.layer_norm(attn_output + x)
        return self.dropout(attn_output)


class LongHorizonBertForTokenClassificationTheForth(BertPreTrainedModel):
    def __init__(self, config):
        super(LongHorizonBertForTokenClassificationTheForth, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.horizon_adapter = HorizonAdapter(config.hidden_size)
        self.capsule_network = CapsuleNetwork(config.hidden_size)
        self.multi_capsule_attention = MultiCapsuleAttention(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
            self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        sequence_output = self.horizon_adapter(sequence_output)
        time_caps_output = self.capsule_network(sequence_output)
        sequence_output = self.multi_capsule_attention(
            sequence_output, time_caps_output
        )
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class HorizonAdapterTheSecond(nn.Module):
    def __init__(self, hidden_size):
        super(HorizonAdapterTheSecond, self).__init__()
        self.feed_forward1 = nn.Linear(hidden_size, hidden_size)
        self.nonlinearity = nn.ReLU()
        self.feed_forward2 = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.nonlinearity(self.feed_forward1(x))
        out = self.feed_forward2(out)
        out = self.layer_norm(out + x)
        return self.dropout(out)


class DepthwiseSeparableConvTheSecond(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, dilation=1):
        super(DepthwiseSeparableConvTheSecond, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return self.relu(out)


class CapsuleNetworkTheSecond(nn.Module):
    def __init__(self, hidden_size):
        super(CapsuleNetworkTheSecond, self).__init__()
        self.conv_features = DepthwiseSeparableConvTheSecond(
            hidden_size, hidden_size, kernel_size=3, padding=1
        )
        self.primary_caps = nn.Linear(hidden_size, hidden_size)
        self.time_caps = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads=8)

    def forward(self, x):
        conv_output = self.conv_features(x.transpose(1, 2))
        primary_caps_output = self.primary_caps(conv_output.transpose(1, 2))
        time_caps_output = self.time_caps(primary_caps_output)
        time_caps_output = self.layer_norm(time_caps_output + x)
        time_caps_output, _ = self.self_attention(
            time_caps_output, time_caps_output, time_caps_output
        )
        return self.dropout(time_caps_output)


class MultiCapsuleAttentionTheSecond(nn.Module):
    def __init__(self, hidden_size):
        super(MultiCapsuleAttentionTheSecond, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, time_caps_output):
        attn_output, _ = self.multi_head_attention(
            x, time_caps_output, time_caps_output
        )
        attn_output = self.layer_norm(attn_output + x)
        return self.dropout(attn_output)


class TransformerEncoderLayerTheSecond(nn.Module):
    def __init__(self, hidden_size):
        super(TransformerEncoderLayerTheSecond, self).__init__()
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attn_output, _ = self.self_attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        return self.dropout(x)


class LongHorizonBertForTokenClassificationTheSecond(BertPreTrainedModel):
    def __init__(self, config):
        super(LongHorizonBertForTokenClassificationTheSecond, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.horizon_adapter = HorizonAdapterTheSecond(config.hidden_size)
        self.capsule_network = CapsuleNetworkTheSecond(config.hidden_size)
        self.multi_capsule_attention = MultiCapsuleAttentionTheSecond(
            config.hidden_size
        )
        self.transformer_encoder = TransformerEncoderLayerTheSecond(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
            self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        sequence_output = self.horizon_adapter(sequence_output)
        time_caps_output = self.capsule_network(sequence_output)
        sequence_output = self.multi_capsule_attention(
            sequence_output, time_caps_output
        )
        sequence_output = self.transformer_encoder(sequence_output)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels, attention_mask)

        return TokenClassifierOutput(loss=loss, logits=logits)

    def compute_loss(self, logits, labels, attention_mask):
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)
        active_labels = torch.where(
            active_loss, labels.view(-1), torch.tensor(-100).type_as(labels)
        )
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        return loss_fct(active_logits, active_labels)


class PrimaryCapsules(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCapsules, self).__init__()
        self.capsules = nn.Conv1d(
            in_channels, num_capsules * out_channels, kernel_size, stride
        )

    def forward(self, x):
        batch_size = x.size(0)
        u = self.capsules(x)
        u = u.view(batch_size, -1, u.size(-1))
        u = self.squash(u)
        return u

    def squash(self, s, epsilon=1e-7):
        s_norm = torch.norm(s, dim=-1, keepdim=True)
        v = (s_norm ** 2 / (1 + s_norm ** 2)) * (s / (s_norm + epsilon))
        return v


class CapsuleNetworkTheThird(nn.Module):
    def __init__(self, input_dim, num_capsules=10, capsule_dim=16, num_iterations=3):
        super(CapsuleNetworkTheThird, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.num_iterations = num_iterations

        self.primary_capsules = PrimaryCapsules(
            num_capsules, input_dim, capsule_dim, kernel_size=1, stride=1
        )
        self.W = nn.Parameter(torch.randn(1, num_capsules, capsule_dim, capsule_dim))

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        u = self.primary_capsules(
            x.permute(0, 2, 1)
        )  # Shape: [batch_size, num_capsules * capsule_dim, seq_len]
        u = u.view(
            batch_size, self.num_capsules, self.capsule_dim, -1
        )  # Shape: [batch_size, num_capsules, capsule_dim, seq_len]
        u = u.permute(
            0, 3, 1, 2
        )  # Shape: [batch_size, seq_len, num_capsules, capsule_dim]

        b = torch.zeros(batch_size, seq_len, self.num_capsules, 1, device=x.device)

        for _ in range(self.num_iterations):
            c = F.softmax(b, dim=2)
            s = (c * u).sum(dim=2, keepdim=True)
            v = self.squash(s)
            b = b + (u * v).sum(dim=-1, keepdim=True)

        v = v.squeeze(2)  # Shape: [batch_size, seq_len, capsule_dim]
        v = v.view(
            batch_size, seq_len, -1
        )  # Shape: [batch_size, seq_len, num_capsules * capsule_dim]
        return v

    def squash(self, s, epsilon=1e-7):
        s_norm = torch.norm(s, dim=-1, keepdim=True)
        v = (s_norm ** 2 / (1 + s_norm ** 2)) * (s / (s_norm + epsilon))
        return v


class MultiCapsuleAttentionTheThird(nn.Module):
    def __init__(self, hidden_size):
        super(MultiCapsuleAttentionTheThird, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, capsule_output):
        attn_output, _ = self.multi_head_attention(x, capsule_output, capsule_output)
        attn_output = self.layer_norm(attn_output + x)
        return self.dropout(attn_output)


class LongHorizonBertForTokenClassificationTheThird(BertPreTrainedModel):
    def __init__(self, config):
        super(LongHorizonBertForTokenClassificationTheThird, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize a list of 48 CapsuleNetworkTheThird instances
        self.capsule_networks = nn.ModuleList(
            [CapsuleNetworkTheThird(config.hidden_size) for _ in range(48)]
        )

        self.multi_capsule_attention = MultiCapsuleAttentionTheThird(config.hidden_size)
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
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # sequence_output = outputs[0]
        sequence_output = outputs[0]
        # print(f"Sequence output shape: {sequence_output.shape}")
        capsule_outputs = []
        for capsule_network in self.capsule_networks:
            capsule_output = capsule_network(sequence_output)
            capsule_outputs.append(capsule_output)

        capsule_output = torch.cat(capsule_outputs, dim=-1)
        sequence_output = self.multi_capsule_attention(sequence_output, capsule_output)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def compute_loss(self, logits, labels, attention_mask):
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)
        active_labels = torch.where(
            active_loss, labels.view(-1), torch.tensor(-100).type_as(labels)
        )
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        return loss_fct(active_logits, active_labels)


class ContextualRelevanceLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ContextualRelevanceLayer, self).__init__()
        self.contextual_embedding = nn.Linear(input_dim, hidden_dim)
        self.relevance_scoring = nn.Linear(hidden_dim, 1)
        self.transform = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Compute contextual embeddings
        context_embeddings = self.contextual_embedding(x)

        # Compute relevance scores
        relevance_scores = self.relevance_scoring(context_embeddings)
        relevance_scores = torch.sigmoid(relevance_scores)

        # Generate contextual mask
        contextual_mask = relevance_scores * context_embeddings

        # Fuse original input with masked context
        fused_sequence = x * contextual_mask

        # Transform the fused sequence
        output = self.transform(fused_sequence)

        return output


from transformers import BertConfig, BertModel, BertPreTrainedModel, PreTrainedModel
from transformers.models.bert.modeling_bert import load_tf_weights_in_bert


class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _supports_sdpa = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# original
class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

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
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TemporalCapsuleNetwork(nn.Module):
    def __init__(self, hidden_size, embed_size=128):
        super(TemporalCapsuleNetwork, self).__init__()
        self.conv_features = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=3, padding=1
        )
        self.primary_caps = nn.Linear(hidden_size, hidden_size)
        self.time_caps = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

        # Date Embeddings
        self.date_embedding = nn.Embedding(
            366, embed_size
        )  # Assume a fixed size for simplicity
        self.linear = nn.Linear(embed_size, hidden_size)

    def forward(self, x, date_indices):
        # Convolutional features
        conv_output = self.conv_features(x.transpose(1, 2))
        primary_caps_output = self.primary_caps(conv_output.transpose(1, 2))

        # Temporal Embedding
        date_emb = self.date_embedding(date_indices)
        date_emb = self.linear(date_emb)

        # Combine capsule output with temporal information
        time_caps_output = self.time_caps(primary_caps_output + date_emb)

        # Layer normalization and dropout
        time_caps_output = self.layer_norm(time_caps_output + x)
        return self.dropout(time_caps_output)


class HorizonAdapter(nn.Module):
    def __init__(self, hidden_size):
        super(HorizonAdapter, self).__init__()
        self.feed_forward1 = nn.Linear(hidden_size, hidden_size)
        self.nonlinearity = nn.ReLU()
        self.feed_forward2 = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.nonlinearity(self.feed_forward1(x))
        out = self.feed_forward2(out)
        out = self.layer_norm(out + x)
        return self.dropout(out)


class MultiCapsuleAttention(nn.Module):
    def __init__(self, hidden_size):
        super(MultiCapsuleAttention, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, time_caps_output):
        attn_output, _ = self.multi_head_attention(
            x, time_caps_output, time_caps_output
        )
        attn_output = self.layer_norm(attn_output + x)
        return self.dropout(attn_output)


class LongHorizonBertForTokenClassificationTemporal(BertPreTrainedModel):
    def __init__(self, config):
        super(LongHorizonBertForTokenClassificationTemporal, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.horizon_adapter = HorizonAdapter(config.hidden_size)
        self.capsule_network = TemporalCapsuleNetwork(config.hidden_size)
        self.multi_capsule_attention = MultiCapsuleAttention(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            date_indices=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        sequence_output = self.horizon_adapter(sequence_output)
        time_caps_output = self.capsule_network(sequence_output, date_indices)
        sequence_output = self.multi_capsule_attention(
            sequence_output, time_caps_output
        )
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def evaluate(args, model, dataset, label_map, prefix="", tokenizer=None):
    # eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
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
    out_sequence_ids, out_token_ids = None, None
    out_sequence_preds, out_token_preds = None, None
    sentences, text_sentences = None, None
    offset_mappings = None
    model.eval()

    # finish = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        # batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch["input_ids"].to(args.device),
                "attention_mask": batch["attention_mask"].to(args.device),
                "labels": batch["token_targets"].to(args.device),
                # "date_indices": batch["date_indices"].to(args.device),
            }

            if isinstance(model, torch.nn.DataParallel):
                model_name = model.module.config._name_or_path
            else:
                model_name = model.config._name_or_path
            if "bert" in model_name and not "modern" in model_name:
                inputs["token_type_ids"] = batch["token_type_ids"].to(args.device)
            elif "xlnet" in model_name:
                inputs["token_type_ids"] = batch["token_type_ids"].to(args.device)

            # Convert tensors to float16 if using fp16
            if args.fp16:
                for key in inputs:
                    inputs[key] = inputs[key].to(torch.float16)

            # model = _move_model_to_device(model, args.device)

            outputs = model(**inputs)

            tokens_result = outputs
            token_logits = outputs.logits
            sequence_logits = None

            tmp_eval_loss = tokens_result.loss

            if args.n_gpu > 1:
                # mean() to average on multi-gpu parallel evaluating
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        if out_token_preds is None:
            # out_token_preds = token_logits.detach().cpu().numpy()
            out_token_preds = token_logits.to(torch.float32).detach().cpu().numpy()

            # out_token_ids = inputs["labels"].detach().cpu().numpy()
            out_token_ids = inputs["labels"].to(torch.float32).detach().cpu().numpy()
            out_input_ids = inputs["input_ids"].to(torch.float32).detach().cpu().numpy()

            sentences = [
                tokenizer.convert_ids_to_tokens(input_ids)
                for input_ids in out_input_ids
            ]
            text_sentences = [text.split(" ") for text in batch["sequence"]]

        else:
            out_token_preds = np.append(
                out_token_preds,
                token_logits.to(torch.float32).detach().cpu().numpy(),
                axis=0,
            )
            out_token_ids = np.append(
                out_token_ids,
                inputs["labels"].to(torch.float32).detach().cpu().numpy(),
                axis=0,
            )

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

            # text_sentences = np.append(text_sentences, [text.split(' ') for text in batch["sequence"]], axis=0)
            text_sentences = text_sentences + [
                text.split(" ") for text in batch["sequence"]
            ]

    out_token_preds = np.argmax(out_token_preds, axis=2)

    logger.info("No sequence classification was performed.")
    report_bin = None
    eval_loss = eval_loss / nb_eval_steps

    label_map = {label: i for i, label in label_map.items()}

    out_label_list = [[] for _ in range(out_token_ids.shape[0])]
    preds_list = [[] for _ in range(out_token_ids.shape[0])]
    words_list = [[] for _ in range(out_token_ids.shape[0])]

    for idx_sentence, item in enumerate(
            zip(text_sentences, out_token_ids, out_token_preds)
    ):
        text_sentence, out_label_ids, out_label_preds = item
        word_ids = tokenizer(text_sentence, is_split_into_words=True).word_ids()
        for idx, word in enumerate(text_sentence):
            beginning_index = word_ids.index(idx)
            try:
                out_label_list[idx_sentence].append(
                    label_map[out_label_ids[beginning_index]]
                )
            except BaseException:  # the sentence was longer then max_length
                out_label_list[idx_sentence].append("O")
            try:
                preds_list[idx_sentence].append(
                    label_map[out_label_preds[beginning_index]]
                )
            except BaseException:  # the sentence was longer then max_length
                preds_list[idx_sentence].append("O")
            words_list[idx_sentence].append(word)

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }

    logger.info("Evaluation for named entity recognition & classification.")
    report_class = seq_classification_report(out_label_list, preds_list, digits=4)
    logger.info("\n%s", report_class)
    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
        wandb.log({f"eval_{key}": results[key]})

    return results, words_list, preds_list, report_bin, report_class


def train(
        args,
        train_dataset,
        dev_dataset,
        test_dataset,
        model,
        tokenizer,
        optimizer,
        label_map,
):
    wandb.watch(
        model, log="all", log_freq=10
    )  # Added log_freq to ensure frequent logging

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

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

    # Prepare optimizer and schedule (linear warmup and decay)
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(
            os.path.join(
                args.output_dir,
                "optimizer.pt",
            )
    ) and os.path.isfile(
        os.path.join(
            args.output_dir,
            "scheduler.pt",
        )
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(
                os.path.join(
                    args.output_dir,
                    "optimizer.pt",
                )
            )
        )
        scheduler.load_state_dict(
            torch.load(
                os.path.join(
                    args.output_dir,
                    "scheduler.pt",
                )
            ),
        )

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

    if getattr(model, "hf_device_map", None) is not None:
        devices = [
            device
            for device in set(model.hf_device_map.values())
            if device not in ["cpu", "disk"]
        ]
        if len(devices) > 1:
            is_model_parallel = True
        elif len(devices) == 1:
            is_model_parallel = args.device != torch.device(devices[0])
        else:
            is_model_parallel = False

        # warn users
        if is_model_parallel:
            logger.info(
                "You have loaded a model on multiple GPUs. `is_model_parallel` attribute will be force-set"
                " to `True` to avoid any unexpected behavior such as device placement mismatching."
            )
        if is_model_parallel:
            place_model_on_device = False

        # Force n_gpu to 1 to avoid DataParallel as MP will manage the GPUs
        if is_model_parallel:
            args._n_gpu = 1

    # multi-gpu training (should be after apex fp16 initialization)
    # Conditionally handle multi-GPU setup
    if args.n_gpu > 1 and not hasattr(model, "hf_device_map"):
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
    # if os.path.exists(args.model_name_or_path):
    #     # set global_step to global_step of last saved checkpoint from model
    #     # path
    #     global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
    #     epochs_trained = global_step // (
    #         len(train_dataloader) // args.gradient_accumulation_steps
    #     )
    #     steps_trained_in_current_epoch = global_step % (
    #         len(train_dataloader) // args.gradient_accumulation_steps
    #     )
    #
    #     logger.info(
    #         "  Continuing training from checkpoint, will skip to saved global_step"
    #     )
    #     logger.info("  Continuing training from epoch %d", epochs_trained)
    #     logger.info("  Continuing training from global step %d", global_step)
    #     logger.info(
    #         "  Will skip the first %d steps in the first epoch",
    #         steps_trained_in_current_epoch,
    #     )

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    set_seed(SEED)  # Added here for reproductibility

    results_devset = {result: [] for result in ["global", "sent-level", "token-level"]}
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
            inputs = {
                "input_ids": batch["input_ids"].to(args.device),
                "attention_mask": batch["attention_mask"].to(args.device),
                "labels": batch["token_targets"].to(args.device),
            }

            # Convert tensors to float16 if using fp16
            if args.fp16:
                for key in inputs:
                    inputs[key] = inputs[key].to(torch.float16)

            if isinstance(model, torch.nn.DataParallel):
                model_name = model.module.config._name_or_path
            else:
                model_name = model.config._name_or_path
            if "bert" in model_name and not "modern" in model_name:
                inputs["token_type_ids"] = batch["token_type_ids"].to(args.device)
            elif "xlnet" in model_name:
                inputs["token_type_ids"] = batch["token_type_ids"].to(args.device)

            outputs = model(**inputs)

            tokens_result = outputs
            loss = tokens_result.loss

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
                    # Log metrics
                    # Only evaluate when single GPU otherwise metrics may not
                    # average well
                    if args.local_rank == -1 and args.evaluate_during_training:

                        results, words_list, preds_list, report_bin, report_class = (
                            evaluate(
                                args, model, dev_dataset, label_map, tokenizer=tokenizer
                            )
                        )

                        write_predictions(
                            args.output_dir,
                            dev_dataset.get_filename(),
                            words_list,
                            preds_list,
                        )

                        for key, value in results.items():
                            wandb.log({f"eval_{key}": value})

                        results_devset["global"].append(results)
                        results_devset["sent-level"].append(report_bin)
                        results_devset["token-level"].append(report_class)

                    wandb.log(
                        {
                            "lr": scheduler.get_lr()[0],
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
                    # model_to_save = (
                    #     model.module if hasattr(model, "module") else model
                    # )  # Take care of distributed/parallel training
                    #
                    # model_to_save.save_pretrained(output_dir)
                    # tokenizer.save_pretrained(output_dir)
                    #
                    # torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    # logger.info("Saving model checkpoint to %s", output_dir)
                    #
                    # torch.save(
                    #     optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                    # )
                    # torch.save(
                    #     scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                    # )
                    # logger.info(
                    #     "Saving optimizer and scheduler states to %s", output_dir
                    # )

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    results, words_list, preds_list, report_bin, report_class = evaluate(
        args, model, test_dataset, label_map, tokenizer=tokenizer
    )

    write_predictions(
        args.output_dir, test_dataset.get_filename(), words_list, preds_list
    )

    results_testset = dict()
    results_testset["global"] = results
    results_testset["token-level"] = report_class

    all_results = {"dev": results_devset, "test": results_testset}
    if "-de" in test_dataset.get_filename():
        with open(os.path.join(args.output_dir, "all_results_de.json"), "w") as f:
            json.dump(all_results, f)
    elif "-fr" in test_dataset.get_filename():
        with open(os.path.join(args.output_dir, "all_results_fr.json"), "w") as f:
            json.dump(all_results, f)
    else:
        logger.info(
            f"Was not able to deduct language from filename of testset, thus no metrics were saved."
        )
    return global_step, tr_loss / global_step


def train_one_epoch(
        args,
        train_dataset,
        dev_dataset,
        test_dataset,
        model,
        tokenizer,
        optimizer,
        scheduler,
        label_map,
        global_step,
        epochs_trained,
        steps_trained_in_current_epoch,
        tr_loss,
        logging_loss,
):
    wandb.watch(
        model, log="all", log_freq=10
    )  # Added log_freq to ensure frequent logging

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

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

    if getattr(model, "hf_device_map", None) is not None:
        devices = [
            device
            for device in set(model.hf_device_map.values())
            if device not in ["cpu", "disk"]
        ]
        if len(devices) > 1:
            is_model_parallel = True
        elif len(devices) == 1:
            is_model_parallel = args.device != torch.device(devices[0])
        else:
            is_model_parallel = False

        # warn users
        if is_model_parallel:
            logger.info(
                "You have loaded a model on multiple GPUs. `is_model_parallel` attribute will be force-set"
                " to `True` to avoid any unexpected behavior such as device placement mismatching."
            )
        if is_model_parallel:
            place_model_on_device = False

        # Force n_gpu to 1 to avoid DataParallel as MP will manage the GPUs
        if is_model_parallel:
            args._n_gpu = 1

    # multi-gpu training (should be after apex fp16 initialization)
    # Conditionally handle multi-GPU setup
    if args.n_gpu > 1 and not hasattr(model, "hf_device_map"):
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    args.train_batch_size = args.train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )

    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    model.zero_grad()
    set_seed(SEED)  # Added here for reproducibility

    results_devset = {result: [] for result in ["global", "sent-level", "token-level"]}

    epoch_iterator = tqdm(
        train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
    )
    for step, batch in enumerate(epoch_iterator):

        # Skip past any already trained steps if resuming training
        if steps_trained_in_current_epoch > 0:
            steps_trained_in_current_epoch -= 1
            continue

        model.train()
        inputs = {
            "input_ids": batch["input_ids"].to(args.device),
            "attention_mask": batch["attention_mask"].to(args.device),
            "labels": batch["token_targets"].to(args.device),
        }

        # Convert tensors to float16 if using fp16
        if args.fp16:
            for key in inputs:
                inputs[key] = inputs[key].to(torch.float16)

        if isinstance(model, torch.nn.DataParallel):
            model_name = model.module.config._name_or_path
        else:
            model_name = model.config._name_or_path
        if "bert" in model_name and not "modern" in model_name:
            inputs["token_type_ids"] = batch["token_type_ids"].to(args.device)
        elif "xlnet" in model_name:
            inputs["token_type_ids"] = batch["token_type_ids"].to(args.device)

        outputs = model(**inputs)

        tokens_result = outputs
        loss = tokens_result.loss

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
            ):
                # Log metrics
                # Only evaluate when single GPU otherwise metrics may not
                # average well
                if args.local_rank == -1 and args.evaluate_during_training:

                    results, words_list, preds_list, report_bin, report_class = (
                        evaluate(
                            args, model, dev_dataset, label_map, tokenizer=tokenizer
                        )
                    )

                    write_predictions(
                        args.output_dir,
                        dev_dataset.get_filename(),
                        words_list,
                        preds_list,
                    )

                    for key, value in results.items():
                        wandb.log({f"eval_{key}": value})

                    results_devset["global"].append(results)
                    results_devset["sent-level"].append(report_bin)
                    results_devset["token-level"].append(report_class)

                wandb.log(
                    {
                        "lr": scheduler.get_lr()[0],
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
                # model_to_save = (
                #     model.module if hasattr(model, "module") else model
                # )  # Take care of distributed/parallel training
                #
                # model_to_save.save_pretrained(output_dir)
                # tokenizer.save_pretrained(output_dir)
                #
                # torch.save(args, os.path.join(output_dir, "training_args.bin"))
                # logger.info("Saving model checkpoint to %s", output_dir)
                #
                # torch.save(
                #     optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                # )
                # torch.save(
                #     scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                # )
                # logger.info("Saving optimizer and scheduler states to %s", output_dir)

        if 0 < args.max_steps < global_step:
            epoch_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    results, words_list, preds_list, report_bin, report_class = evaluate(
        args, model, test_dataset, label_map, tokenizer=tokenizer
    )

    write_predictions(
        args.output_dir, test_dataset.get_filename(), words_list, preds_list
    )

    results_testset = dict()
    results_testset["global"] = results
    results_testset["token-level"] = report_class

    all_results = {"dev": results_devset, "test": results_testset}
    if "-de" in test_dataset.get_filename():
        with open(os.path.join(args.output_dir, "all_results_de.json"), "w") as f:
            json.dump(all_results, f)
    elif "-fr" in test_dataset.get_filename():
        with open(os.path.join(args.output_dir, "all_results_fr.json"), "w") as f:
            json.dump(all_results, f)
    else:
        logger.info(
            f"Was not able to deduct language from filename of testset, thus no metrics were saved."
        )
    return global_step, tr_loss / global_step, logging_loss

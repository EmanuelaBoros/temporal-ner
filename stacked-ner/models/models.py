# -*- coding: utf-8 -*-

from crf import ConditionalRandomField, allowed_transitions
from transformer import TransformerEncoder, MultiHeadAttn, TransformerLayer
import torch.nn.functional as F

import torch
import torch.nn as nn
import logging
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StackedTransformersCRF(nn.Module):
    def __init__(self, tag_vocabs, embed, embed_doc, num_layers, d_model, n_head, feedforward_dim, dropout,
                 after_norm=True, attn_type='adatrans', bi_embed=None,
                 fc_dropout=0.3, pos_embed=None, scale=False, dropout_attn=None):

        super().__init__()

        self.embed = embed
        self.embed_doc = embed_doc

        embed_size = self.embed.embed_size
        self.bi_embed = None
        if bi_embed is not None:
            self.bi_embed = bi_embed
            embed_size += self.bi_embed.embed_size

        self.tag_vocabs = []
        self.out_fcs = nn.ModuleList()
        self.crfs = nn.ModuleList()

        for i in range(len(tag_vocabs)):
            self.tag_vocabs.append(tag_vocabs[i])

            #            linear = nn.Linear(768, len(tag_vocabs[i]))
            linear = nn.Linear(1536, len(tag_vocabs[i]))
            #            linear = nn.Linear(1792, len(tag_vocabs[i]))

            self.out_fcs.append(linear)
            trans = allowed_transitions(
                tag_vocabs[i], encoding_type='bioes', include_start_end=True)
            crf = ConditionalRandomField(
                len(tag_vocabs[i]), include_start_end_trans=True, allowed_transitions=trans)
            self.crfs.append(crf)

        self.in_fc = nn.Linear(embed_size, d_model)

        self.transformer = TransformerEncoder(num_layers, d_model, n_head, feedforward_dim, dropout,
                                              after_norm=after_norm, attn_type=attn_type,
                                              scale=scale, dropout_attn=dropout_attn,
                                              pos_embed=pos_embed)

        self.transformer_doc = TransformerEncoder(num_layers, d_model, n_head, feedforward_dim, dropout,
                                                  after_norm=after_norm, attn_type=attn_type,
                                                  scale=scale, dropout_attn=dropout_attn,
                                                  pos_embed=pos_embed)
        self.self_attn = MultiHeadAttn(d_model, n_head)

        self.pooling_methods = ['max', 'mean', 'max-mean']

        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc_dropout_doc = nn.Dropout(fc_dropout)

    def _forward(self, words, doc=None, target=None, target1=None, target2=None, target3=None,
                 target4=None, target5=None, target6=None, bigrams=None, seq_len=None):

        torch.cuda.empty_cache()

        mask = words.ne(0)
        words = self.embed(words)

        torch.cuda.empty_cache()
        targets = [target, target1, target2, target3, target4, target5]

        chars = self.in_fc(words)
        chars = self.transformer(chars, mask)
        words = self.fc_dropout(chars)

        logits = []
        for i in range(len(targets)):
            logits.append(F.log_softmax(self.out_fcs[i](words), dim=-1))

        torch.cuda.empty_cache()

        if target is not None:
            losses = []
            for i in range(len(targets)):
                losses.append(self.crfs[i](logits[i], targets[i], mask))

            return {'loss': sum(losses)}
        else:
            results = {}
            for i in range(len(targets)):
                confidence = torch.nn.functional.softmax(logits[i], dim=2)
                # confidence = np.max(confidence.detach().cpu().numpy(), axis=2)[0]
                if i == 0:
                    results['pred'] = self.crfs[i].viterbi_decode(logits[i], mask)[0]
                    results['confidence'] = confidence
                else:  # torch.argmax(logits[i], 2) #
                    results['pred' + str(i)] = self.crfs[i].viterbi_decode(logits[i], mask)[0]
                    results['confidence' + str(i)] = confidence

            return results

    def forward(self, words, doc=None, target=None, target1=None, target2=None,
                target3=None, target4=None, target5=None, target6=None, seq_len=None):
        return self._forward(words, doc, target, target1, target2,
                             target3, target4, target5, target6, seq_len)

    def predict(self, words, doc=None, seq_len=None):
        return self._forward(words, doc, target=None)

    def save(self, file_path, _extra_files=None):
        """
        Saves the model state to the specified file path.

        Args:
        file_path (str): The file path where the model state will be saved.
        """
        torch.save(self.state_dict(), file_path)


class BertCRF(nn.Module):
    def __init__(self, embed, tag_vocabs, encoding_type='bio'):
        super().__init__()
        self.embed = embed
        self.tag_vocabs = []
        self.fcs = nn.ModuleList()
        self.crfs = nn.ModuleList()

        for i in range(len(tag_vocabs)):
            self.tag_vocabs.append(tag_vocabs[i])
            linear = nn.Linear(self.embed.embed_size, len(tag_vocabs[i]))
            self.fcs.append(linear)
            trans = allowed_transitions(
                tag_vocabs[i], encoding_type=encoding_type, include_start_end=True)
            crf = ConditionalRandomField(
                len(tag_vocabs[i]), include_start_end_trans=True, allowed_transitions=trans)
            self.crfs.append(crf)

    def _forward(self, words, target=None, target1=None, target2=None, target3=None,
                 target4=None, target5=None, target6=None, seq_len=None):

        # logger.info(f'words in models: {words}')
        mask = words.ne(0)
        words = self.embed(words)

        targets = [target, target1, target2, target3, target4, target5]

        words_fcs = []
        for i in range(len(targets)):
            # import pdb;pdb.set_trace()
            words_fcs.append(self.fcs[i](words))

        logits = []
        for i in range(len(targets)):
            logits.append(F.log_softmax(words_fcs[i], dim=-1))

        if target is not None:
            losses = []
            for i in range(len(targets)):
                losses.append(self.crfs[i](logits[i], targets[i], mask))

            return {'loss': sum(losses)}
        else:
            results = {}
            for i in range(len(targets)):
                confidence = torch.nn.functional.softmax(logits[i], dim=2)
                confidence = np.max(confidence.detach().cpu().numpy(), axis=2)[0]
                if i == 0:
                    results['pred'] = self.crfs[i].viterbi_decode(logits[i], mask)[0]
                    results['confidence'] = confidence
                else:
                    results['pred' + str(i)] = self.crfs[i].viterbi_decode(logits[i], mask)[0]
                    # torch.argmax(logits[i], 2)
                    results['confidence' + str(i)] = confidence

            return results

    """

    """

    def forward(self, words, target=None, target1=None, target2=None, target3=None, target4=None, target5=None,
                target6=None, seq_len=None):
        return self._forward(words, target, target1, target2, target3, target4, target5, target6, seq_len)

    def predict(self, words, seq_len=None):
        return self._forward(words, target=None)

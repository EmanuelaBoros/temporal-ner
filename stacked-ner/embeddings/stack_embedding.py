r"""
.. todo::
    doc
"""

__all__ = [
    "StackEmbedding",
]

from typing import List

import torch
from torch import nn as nn
from abc import ABCMeta, abstractmethod
from .embedding import TokenEmbedding


def _check_vocab_has_same_index(vocab, other_vocab):
    if other_vocab != vocab:
        for word, word_ix in vocab:
            other_word_idx = other_vocab.to_index(word)
            assert other_word_idx == word_ix, f"Word {word} has different index in vocabs, {word_ix} Vs. {other_word_idx}."


class TokenEmbedding(nn.Module):

    def __init__(self, vocab, word_dropout=0.0, dropout=0.0):
        super(TokenEmbedding, self).__init__()
        if vocab.rebuild:
            vocab.build_vocab()
        assert vocab.padding is not None, "Vocabulary must have a padding entry."
        self._word_vocab = vocab
        self._word_pad_index = vocab.padding_idx
        if word_dropout > 0:
            assert vocab.unknown is not None, "Vocabulary must have unknown entry when you want to drop a word."
        self.word_dropout = word_dropout
        self._word_unk_index = vocab.unknown_idx
        self.dropout_layer = nn.Dropout(dropout)

    def drop_word(self, words):
        if self.word_dropout > 0 and self.training:
            mask = torch.full_like(words, fill_value=self.word_dropout, dtype=torch.float, device=words.device)
            mask = torch.bernoulli(mask).eq(1)
            pad_mask = words.ne(self._word_pad_index)
            mask = mask.__and__(pad_mask)
            words = words.masked_fill(mask, self._word_unk_index)
        return words

    def dropout(self, words):

        return self.dropout_layer(words)

    @property
    def requires_grad(self):

        requires_grads = set([param.requires_grad for param in self.parameters()])
        if len(requires_grads) == 1:
            return requires_grads.pop()
        else:
            return None

    @requires_grad.setter
    def requires_grad(self, value):
        for param in self.parameters():
            param.requires_grad = value

    def __len__(self):
        return len(self._word_vocab)

    @property
    def embed_size(self) -> int:
        return self._embed_size

    @property
    def embedding_dim(self) -> int:
        return self._embed_size

    @property
    def num_embedding(self) -> int:

        return len(self._word_vocab)

    def get_word_vocab(self):

        return self._word_vocab

    @property
    def size(self):
        return torch.Size((self.num_embedding, self._embed_size))

    @abstractmethod
    def forward(self, words):
        raise NotImplementedError

class StackEmbedding(TokenEmbedding):

    def __init__(self, embeds: List[TokenEmbedding], word_dropout=0, dropout=0):

        vocabs = []
        for embed in embeds:
            if hasattr(embed, 'get_word_vocab'):
                vocabs.append(embed.get_word_vocab())
        _vocab = vocabs[0]
        for vocab in vocabs[1:]:
            if _vocab != vocab:
                _check_vocab_has_same_index(_vocab, vocab)

        super(StackEmbedding, self).__init__(_vocab, word_dropout=word_dropout, dropout=dropout)
        assert isinstance(embeds, list)
        # for embed in embeds:
        #     assert isinstance(embed, TokenEmbedding), "Only TokenEmbedding type is supported."
        self.embeds = nn.ModuleList(embeds)
        self._embed_size = sum([embed.embed_size for embed in self.embeds])

    def append(self, embed: TokenEmbedding):
        _check_vocab_has_same_index(self.get_word_vocab(), embed.get_word_vocab())
        self._embed_size += embed.embed_size
        self.embeds.append(embed)
        return self

    def pop(self):
        embed = self.embeds.pop()
        self._embed_size -= embed.embed_size
        return embed

    @property
    def embed_size(self):

        return self._embed_size

    def forward(self, words):

        outputs = []
        words = self.drop_word(words)
        for embed in self.embeds:
            outputs.append(embed(words))
        outputs = self.dropout(torch.cat(outputs, dim=-1))
        return outputs

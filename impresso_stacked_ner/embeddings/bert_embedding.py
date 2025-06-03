# -*- coding: utf-8 -*-
"""https://github.com/fastnlp/fastNLP"""
__all__ = [
    "BertEmbedding",
]

from itertools import chain
import logging
import numpy as np
from torch import nn
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from transformers import AutoTokenizer, AutoModel
from abc import abstractmethod

try:
    from .embedding import TokenEmbedding
except:
    from embedding import TokenEmbedding
from fastNLP.core import logger
from fastNLP.core.batch import DataSetIter
from fastNLP.core.dataset import DataSet
from fastNLP.core.sampler import SequentialSampler
from fastNLP.core.utils import _move_model_to_device, _get_model_device
from fastNLP.core.vocabulary import Vocabulary


class TokenEmbedding(nn.Module):

    def __init__(self, vocab, word_dropout=0.0, dropout=0.0):
        super(TokenEmbedding, self).__init__()
        if vocab.rebuild:
            vocab.build_vocab()
        assert vocab.padding is not None, "Vocabulary must have a padding entry."
        self._word_vocab = vocab
        self._word_pad_index = vocab.padding_idx
        if word_dropout > 0:
            assert (
                vocab.unknown is not None
            ), "Vocabulary must have unknown entry when you want to drop a word."
        self.word_dropout = word_dropout
        self._word_unk_index = vocab.unknown_idx
        self.dropout_layer = nn.Dropout(dropout)

    def drop_word(self, words):
        if self.word_dropout > 0 and self.training:
            mask = torch.full_like(
                words,
                fill_value=self.word_dropout,
                dtype=torch.float,
                device=words.device,
            )
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


class ContextualEmbedding(TokenEmbedding):

    def __init__(
        self, vocab: Vocabulary, word_dropout: float = 0.0, dropout: float = 0.0
    ):
        super(ContextualEmbedding, self).__init__(
            vocab, word_dropout=word_dropout, dropout=dropout
        )

    def add_sentence_cache(
        self, *datasets, batch_size=32, device="cpu", delete_weights: bool = True
    ):

        for index, dataset in enumerate(datasets):
            try:
                assert isinstance(
                    dataset, DataSet
                ), "Only fastNLP.DataSet object is allowed."
                assert (
                    "words" in dataset.get_input_name()
                ), "`words` field has to be set as input."
            except Exception as e:
                logger.error(f"Exception happens at {index} dataset.")
                raise e

        sent_embeds = {}
        _move_model_to_device(self, device=device)
        device = _get_model_device(self)
        pad_index = self._word_vocab.padding_idx
        logger.info("Start to calculate sentence representations.")
        with torch.no_grad():
            for index, dataset in enumerate(datasets):
                try:
                    batch = DataSetIter(
                        dataset, batch_size=batch_size, sampler=SequentialSampler()
                    )
                    for batch_x, batch_y in batch:
                        words = batch_x["words"].to(device)
                        words_list = words.tolist()
                        seq_len = words.ne(pad_index).sum(dim=-1)
                        max_len = words.size(1)
                        seq_len_from_behind = (max_len - seq_len).tolist()
                        word_embeds = self(words).detach().cpu().numpy()
                        for b in range(words.size(0)):
                            length = seq_len_from_behind[b]
                            if length == 0:
                                sent_embeds[tuple(words_list[b][: seq_len[b]])] = (
                                    word_embeds[b]
                                )
                            else:
                                sent_embeds[tuple(words_list[b][: seq_len[b]])] = (
                                    word_embeds[b, :-length]
                                )
                except Exception as e:
                    logger.error(f"Exception happens at {index} dataset.")
                    raise e
        logger.info("Finish calculating sentence representations.")
        self.sent_embeds = sent_embeds
        if delete_weights:
            self._delete_model_weights()

    def _get_sent_reprs(self, words):

        if hasattr(self, "sent_embeds"):
            words_list = words.tolist()
            seq_len = words.ne(self._word_pad_index).sum(dim=-1)
            _embeds = []
            for b in range(len(words)):
                words_i = tuple(words_list[b][: seq_len[b]])
                embed = self.sent_embeds[words_i]
                _embeds.append(embed)
            max_sent_len = max(map(len, _embeds))
            embeds = words.new_zeros(
                len(_embeds),
                max_sent_len,
                self.embed_size,
                dtype=torch.float,
                device=words.device,
            )
            for i, embed in enumerate(_embeds):
                embeds[i, : len(embed)] = torch.FloatTensor(embed).to(words.device)
            return embeds
        return None

    @abstractmethod
    def _delete_model_weights(self):
        raise NotImplementedError

    def remove_sentence_cache(self):
        del self.sent_embeds


class BertEmbedding(ContextualEmbedding):

    def __init__(
        self,
        vocab: Vocabulary,
        model_dir_or_name: str = "bert-base-uncased",
        layers: str = "-1",
        pool_method: str = "first",
        word_dropout=0,
        dropout=0,
        include_cls_sep: bool = False,
        pooled_cls=True,
        requires_grad: bool = True,
        auto_truncate: bool = False,
        **kwargs,
    ):

        super(BertEmbedding, self).__init__(
            vocab, word_dropout=word_dropout, dropout=dropout
        )

        self.vocab = vocab

        if word_dropout > 0:
            assert (
                vocab.unknown is not None
            ), "When word_drop>0, Vocabulary must contain the unknown token."

        only_use_pretrain_bpe = kwargs.get("only_use_pretrain_bpe", False)
        truncate_embed = kwargs.get("truncate_embed", True)
        min_freq = kwargs.get("min_freq", 2)

        logger.debug("Finish tokenizing words to word pieces.")

        encoder = AutoModel.from_pretrained(model_dir_or_name, torchscript=True)

        # encoder = torch.jit.trace(AutoModel.from_pretrained(model_dir_or_name, torchscript=True),
        #                 torch.rand(1, 512))

        tokenizer = AutoTokenizer.from_pretrained(model_dir_or_name, torchscript=True)

        self.model = _BertWordModel(
            model_dir_or_name=model_dir_or_name,
            vocab=vocab,
            layers=layers,
            pool_method=pool_method,
            include_cls_sep=include_cls_sep,
            pooled_cls=pooled_cls,
            auto_truncate=auto_truncate,
            min_freq=min_freq,
            only_use_pretrain_bpe=only_use_pretrain_bpe,
            truncate_embed=truncate_embed,
            tokenizer=tokenizer,
            encoder=encoder,
        )

        self.requires_grad = requires_grad
        self._embed_size = (
            len(self.model.layers) * self.model.encoder.config.hidden_size
        )

        self._word_sep_index = self.model._sep_index
        self._word_pad_index = self.model._word_pad_index
        self._word_cls_index = self.model._cls_index

    def _delete_model_weights(self):
        del self.model

    def forward(self, words):

        # words = self.drop_word(words)
        outputs = self._get_sent_reprs(words)
        if outputs is not None:
            return self.dropout(outputs)
        outputs = self.model(words)
        outputs = torch.cat([*outputs], dim=-1)
        return self.dropout(outputs)

    def drop_word(self, words):
        if self.word_dropout > 0 and self.training:
            with torch.no_grad():
                mask = torch.full_like(
                    words,
                    fill_value=self.word_dropout,
                    dtype=torch.float,
                    device=words.device,
                )
                mask = torch.bernoulli(mask).eq(1)
                pad_mask = words.ne(self._word_pad_index)
                mask = pad_mask.__and__(mask)
                if self._word_sep_index != -100:
                    not_sep_mask = words.ne(self._word_sep_index)
                    mask = mask.__and__(not_sep_mask)
                if self._word_cls_index != -100:
                    not_cls_mask = words.ne(self._word_cls_index)
                    mask = mask.__and__(not_cls_mask)
                words = words.masked_fill(mask, self._word_unk_index)
        return words


class _BertWordModel(nn.Module):
    def __init__(
        self,
        model_dir_or_name: str,
        vocab: Vocabulary,
        layers: str = "-1",
        pool_method: str = "first",
        include_cls_sep: bool = False,
        pooled_cls: bool = False,
        auto_truncate: bool = False,
        tokenizer: AutoTokenizer = None,
        encoder: AutoModel = None,
        min_freq=2,
        only_use_pretrain_bpe=False,
        truncate_embed=True,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.encoder = encoder

        # import pdb;pdb.set_trace()
        self._max_position_embeddings = self.encoder.config.max_position_embeddings
        self.encoder_layer_number = self.encoder.config.num_hidden_layers

        if isinstance(layers, list):
            self.layers = [int(l) for l in layers]
        elif isinstance(layers, str):
            self.layers = list(map(int, layers.split(",")))
        else:
            raise TypeError("`layers` only supports str or list[int]")
        for layer in self.layers:
            if layer < 0:
                assert -layer <= self.encoder_layer_number, (
                    f"The layer index:{layer} is out of scope for "
                    f"a bert model with {self.encoder_layer_number} layers."
                )
            else:
                assert layer <= self.encoder_layer_number, (
                    f"The layer index:{layer} is out of scope for "
                    f"a bert model with {self.encoder_layer_number} layers."
                )

        assert pool_method in ("avg", "max", "first", "last")

        self.pool_method = pool_method
        self.include_cls_sep = include_cls_sep
        self.pooled_cls = pooled_cls
        self.auto_truncate = auto_truncate

        # logger.info("Vocab is not used anymore.")
        # logger.info("Start to generate word pieces for word.")
        self._has_sep_in_vocab = "[SEP]" in vocab

        word_to_wordpieces = []
        word_pieces_lengths = []
        for word, index in vocab:
            if index == vocab.padding_idx:
                word = self.tokenizer.pad_token
            elif index == vocab.unknown_idx:
                word = self.tokenizer.unk_token

            word_pieces = self.tokenizer.tokenize(word)
            # just tokenize in sub-words
            # import pdb;pdb.set_trace()
            word_pieces = self.tokenizer.convert_tokens_to_ids(word_pieces)
            word_to_wordpieces.append(word_pieces)
            word_pieces_lengths.append(len(word_pieces))

        print("Vocabulary length:", len(vocab))
        self._cls_index = self.tokenizer.cls_token_id
        if self._cls_index is None:
            self._cls_index = self.tokenizer.bos_token_id

        self._sep_index = self.tokenizer.sep_token_id
        if self._sep_index is None:
            self._sep_index = self.tokenizer.eos_token_id

        self._word_pad_index = vocab.padding_idx
        self._wordpiece_pad_index = self.tokenizer.pad_token_id
        if self._wordpiece_pad_index is None:
            self._wordpiece_pad_index = 0

        max_wordpiece_length = max(word_pieces_lengths)
        # Pad every sequence in word_to_wordpieces to max_wordpiece_length self.encoder.config.max_position_embeddings #
        padded_word_to_wordpieces = []
        for word_pieces in word_to_wordpieces:
            padded_word_pieces = word_pieces + [self._wordpiece_pad_index] * (
                max_wordpiece_length - len(word_pieces)
            )
            padded_word_to_wordpieces.append(padded_word_pieces)

        # TODO: these are the shapes of the words which later should not contain 0s
        self.word_to_wordpieces = np.array(padded_word_to_wordpieces)
        self.register_buffer(
            "word_pieces_lengths", torch.LongTensor(word_pieces_lengths)
        )
        logger.debug("Successfully generate word pieces.")

        self.vocab = vocab

    # @torch.jit.ignore
    # @property
    def transform_word_to_bert_inputs(self, words):

        batch_size, max_word_len = words.size()
        word_mask = words.ne(self._word_pad_index)
        seq_len = word_mask.sum(dim=-1)
        # Assuming word_mask is a boolean tensor with True/False values
        word_mask = word_mask.int()

        batch_word_pieces_length = self.word_pieces_lengths[words].masked_fill(
            word_mask.eq(int(False)), 0
        )

        word_pieces_lengths = batch_word_pieces_length.sum(dim=-1)
        # TODO: repaired here the lengths of the attentions in an unorthodox way

        word_pieces_lengths = torch.clamp(seq_len, max=word_pieces_lengths[0])

        max_word_piece_length = batch_word_pieces_length.sum(dim=-1).max().item()
        if max_word_piece_length + 2 > self._max_position_embeddings:
            if self.auto_truncate:
                word_pieces_lengths = word_pieces_lengths.masked_fill(
                    word_pieces_lengths + 2 > self._max_position_embeddings,
                    self._max_position_embeddings - 2,
                )
            else:
                raise RuntimeError(
                    "After split words into word pieces, the lengths of word pieces are longer than the "
                    f"maximum allowed sequence length:{self._max_position_embeddings} of bert. You can set "
                    f"`auto_truncate=True` for BertEmbedding to automatically truncate overlong input."
                )

        word_pieces = words.new_full(
            (batch_size, min(max_word_piece_length + 2, self._max_position_embeddings)),
            fill_value=0,
        )

        attn_masks = torch.zeros_like(word_pieces)
        word_indexes = words.cpu().numpy()

        for i in range(batch_size):
            word_pieces_i = []
            for x in chain(*self.word_to_wordpieces[word_indexes[i, :]]):
                if x not in [self._wordpiece_pad_index, str(self._wordpiece_pad_index)]:
                    word_pieces_i.append(x)

            if (
                self.auto_truncate
                and len(word_pieces_i) > self._max_position_embeddings - 2
            ):
                word_pieces_i = word_pieces_i[: self._max_position_embeddings - 2]

            word_pieces[i, : len(word_pieces_i)] = torch.LongTensor(
                word_pieces_i[: len(word_pieces_i)]
            )

            # TODO: be careful here: it was before torch.LongTensor(word_pieces_i))
            attn_masks[i, : word_pieces_lengths[i] + 2].fill_(1)

        word_pieces[:, 0].fill_(self._cls_index)
        batch_indexes = torch.arange(batch_size).to(words)
        # import pdb;pdb.set_trace()
        word_pieces[batch_indexes, word_pieces_lengths + 1] = self._sep_index

        return (
            batch_size,
            max_word_len,
            seq_len,
            batch_word_pieces_length,
            word_mask,
            max_word_piece_length,
            batch_indexes,
            word_pieces,
            attn_masks,
            word_pieces_lengths,
        )

    def forward(self, words):

        with torch.no_grad():

            (
                batch_size,
                max_word_len,
                seq_len,
                batch_word_pieces_length,
                word_mask,
                max_word_piece_length,
                batch_indexes,
                word_pieces,
                attn_masks,
                word_pieces_lengths,
            ) = self.transform_word_to_bert_inputs(words)

            if self._has_sep_in_vocab:
                sep_mask = word_pieces.eq(self._sep_index).long()
                sep_mask_cumsum = (
                    sep_mask.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
                )
                token_type_ids = sep_mask_cumsum.fmod(2)
                if token_type_ids[0, 0].item():
                    token_type_ids = token_type_ids.eq(0).long()
            else:
                token_type_ids = torch.zeros_like(word_pieces)

        logger.debug("Finish tokenizing words to word pieces.")
        # print('word_pieces:', word_pieces.shape)
        max_length = 512
        # # Truncate word_pieces, attn_masks, and token_type_ids to max_length
        word_pieces = word_pieces[:, :max_length]
        attn_masks = attn_masks[:, :max_length]
        token_type_ids = (
            token_type_ids[:, :max_length]
            if self._has_sep_in_vocab
            else torch.zeros_like(word_pieces)
        )

        item = self.encoder(
            input_ids=word_pieces,
            token_type_ids=token_type_ids,
            attention_mask=attn_masks,
            output_hidden_states=True,
        )

        try:
            _, pooled_cls, bert_outputs = item[0], item[1], item[2]
        except:
            pooled_cls, bert_outputs = item[0], item[1]

        if self.include_cls_sep:
            s_shift = 1
            outputs = bert_outputs[-1].new_zeros(
                len(self.layers),
                batch_size,
                max_word_len + 2,
                bert_outputs[-1].size(-1),
            )

        else:
            s_shift = 0
            outputs = bert_outputs[-1].new_zeros(
                len(self.layers), batch_size, max_word_len, bert_outputs[-1].size(-1)
            )

        batch_word_pieces_cum_length = batch_word_pieces_length.new_zeros(
            batch_size, max_word_len + 1
        )
        batch_word_pieces_cum_length[:, 1:] = batch_word_pieces_length.cumsum(dim=-1)

        if self.pool_method == "first":
            batch_word_pieces_cum_length = batch_word_pieces_cum_length[
                :, : seq_len.max()
            ]
            batch_word_pieces_cum_length.masked_fill_(
                batch_word_pieces_cum_length.ge(max_word_piece_length), 0
            )
            _batch_indexes = batch_indexes[:, None].expand(
                (batch_size, batch_word_pieces_cum_length.size(1))
            )
        elif self.pool_method == "last":
            batch_word_pieces_cum_length = (
                batch_word_pieces_cum_length[:, 1 : seq_len.max() + 1] - 1
            )
            batch_word_pieces_cum_length.masked_fill_(
                batch_word_pieces_cum_length.ge(max_word_piece_length), 0
            )
            _batch_indexes = batch_indexes[:, None].expand(
                (batch_size, batch_word_pieces_cum_length.size(1))
            )

        for l_index, l in enumerate(self.layers):
            output_layer = bert_outputs[l]
            real_word_piece_length = output_layer.size(1) - 2
            if max_word_piece_length > real_word_piece_length:
                paddings = output_layer.new_zeros(
                    batch_size,
                    max_word_piece_length - real_word_piece_length,
                    output_layer.size(2),
                )
                output_layer = torch.cat((output_layer, paddings), dim=1).contiguous()
            truncate_output_layer = output_layer[:, 1:-1]
            if self.pool_method == "first":
                tmp = truncate_output_layer[
                    _batch_indexes, batch_word_pieces_cum_length
                ]
                tmp = tmp.int()
                tmp = tmp.masked_fill(
                    word_mask[:, : batch_word_pieces_cum_length.size(1), None].eq(
                        int(False)
                    ),
                    0,
                )
                outputs[
                    l_index, :, s_shift : batch_word_pieces_cum_length.size(1) + s_shift
                ] = tmp

            elif self.pool_method == "last":
                tmp = truncate_output_layer[
                    _batch_indexes, batch_word_pieces_cum_length
                ]
                tmp = tmp.int()
                tmp = tmp.masked_fill(
                    word_mask[:, : batch_word_pieces_cum_length.size(1), None].eq(
                        int(False)
                    ),
                    0,
                )
                outputs[
                    l_index, :, s_shift : batch_word_pieces_cum_length.size(1) + s_shift
                ] = tmp
            elif self.pool_method == "max":
                for i in range(batch_size):
                    for j in range(seq_len[i]):
                        start, end = (
                            batch_word_pieces_cum_length[i, j],
                            batch_word_pieces_cum_length[i, j + 1],
                        )
                        outputs[l_index, i, j + s_shift], _ = torch.max(
                            truncate_output_layer[i, start:end], dim=-2
                        )
            else:
                for i in range(batch_size):
                    for j in range(seq_len[i]):
                        start, end = (
                            batch_word_pieces_cum_length[i, j],
                            batch_word_pieces_cum_length[i, j + 1],
                        )
                        outputs[l_index, i, j + s_shift] = torch.mean(
                            truncate_output_layer[i, start:end], dim=-2
                        )
            if self.include_cls_sep:
                if l in (len(bert_outputs) - 1, -1) and self.pooled_cls:
                    outputs[l_index, :, 0] = pooled_cls
                else:
                    outputs[l_index, :, 0] = output_layer[:, 0]
                outputs[l_index, batch_indexes, seq_len + s_shift] = output_layer[
                    batch_indexes, word_pieces_lengths + s_shift
                ]

        return outputs

from transformers import Pipeline
import numpy as np
import torch
from nltk.chunk import conlltags2tree
from nltk import pos_tag
from nltk.tree import Tree
import string
import torch.nn.functional as F
from langdetect import detect


import re, string
import pysbd


def tokenize(text):
    # print(text)
    for punctuation in string.punctuation:
        text = text.replace(punctuation, " " + punctuation + " ")
    return text.split()


def normalize_text(text):
    # Remove spaces and tabs for the search but keep newline characters
    return re.sub(r"[ \t]+", "", text)


def find_entity_indices(article_text, search_text):
    # Normalize texts by removing spaces and tabs
    normalized_article = normalize_text(article_text)
    normalized_search = normalize_text(search_text)

    # Initialize a list to hold all start and end indices
    indices = []

    # Find all occurrences of the search text in the normalized article text
    start_index = 0
    while True:
        start_index = normalized_article.find(normalized_search, start_index)
        if start_index == -1:
            break

        # Calculate the actual start and end indices in the original article text
        original_chars = 0
        original_start_index = 0
        for i in range(start_index):
            while article_text[original_start_index] in (" ", "\t"):
                original_start_index += 1
            if article_text[original_start_index] not in (" ", "\t", "\n"):
                original_chars += 1
            original_start_index += 1

        original_end_index = original_start_index
        search_chars = 0
        while search_chars < len(normalized_search):
            if article_text[original_end_index] not in (" ", "\t", "\n"):
                search_chars += 1
            original_end_index += 1  # Increment to include the last character

        # Append the found indices to the list
        if article_text[original_start_index] == " ":
            original_start_index += 1
        indices.append((original_start_index, original_end_index))

        # Move start_index to the next position to continue searching
        start_index += 1

    return indices


def get_entities(tokens, tags, confidences, text):

    tags = [tag.replace("S-", "B-").replace("E-", "I-") for tag in tags]
    pos_tags = [pos for token, pos in pos_tag(tokens)]

    conlltags = [(token, pos, tg) for token, pos, tg in zip(tokens, pos_tags, tags)]
    ne_tree = conlltags2tree(conlltags)

    entities = []
    idx: int = 0
    already_done = []
    for subtree in ne_tree:
        # skipping 'O' tags
        if isinstance(subtree, Tree):
            original_label = subtree.label()
            original_string = " ".join([token for token, pos in subtree.leaves()])

            for indices in find_entity_indices(text, original_string):
                entity_start_position = indices[0]
                entity_end_position = indices[1]
                if (
                    "_".join(
                        [original_label, original_string, str(entity_start_position)]
                    )
                    in already_done
                ):
                    continue
                else:
                    already_done.append(
                        "_".join(
                            [
                                original_label,
                                original_string,
                                str(entity_start_position),
                            ]
                        )
                    )
                entities.append(
                    {
                        "entity": original_label,
                        "score": np.average(confidences[idx : idx + len(subtree)])
                        * 100,
                        "index": (idx, idx + len(subtree)),
                        "word": original_string,
                        "start": entity_start_position,
                        "end": entity_end_position,
                    }
                )

            idx += len(subtree)

            # Update the current character position
            # We add the length of the original string + 1 (for the space)
        else:
            token, pos = subtree
            # If it's not a named entity, we still need to update the character
            # position
            idx += 1

    return entities


def realign(
    text_sentence, out_label_preds, softmax_scores, tokenizer, reverted_label_map
):
    preds_list, words_list, confidence_list = [], [], []
    word_ids = tokenizer(text_sentence, is_split_into_words=True).word_ids()
    for idx, word in enumerate(text_sentence):
        beginning_index = word_ids.index(idx)
        try:
            preds_list.append(reverted_label_map[out_label_preds[beginning_index]])
            confidence_list.append(max(softmax_scores[beginning_index]))
        except Exception as ex:  # the sentence was longer then max_length
            preds_list.append("O")
            confidence_list.append(0.0)
        words_list.append(word)

    return words_list, preds_list, confidence_list


def segment_and_trim_sentences(article, language, max_length):

    try:
        segmenter = pysbd.Segmenter(language=language, clean=False)
    except:
        segmenter = pysbd.Segmenter(language="en", clean=False)

    sentences = segmenter.segment(article)

    trimmed_sentences = []
    for sentence in sentences:
        while len(sentence) > max_length:
            # Find the last space within max_length
            cut_index = sentence.rfind(" ", 0, max_length)
            if cut_index == -1:
                # If no space found, forcibly cut at max_length
                cut_index = max_length

            # Cut the sentence and add the first part to trimmed sentences
            trimmed_sentences.append(sentence[:cut_index])

            # Update the sentence to be the remaining part
            sentence = sentence[cut_index:].lstrip()

        # Add the remaining part of the sentence if it's not empty
        if sentence:
            trimmed_sentences.append(sentence)

    return trimmed_sentences


# List of additional "strange" punctuation marks
additional_punctuation = "‘’“”„«»•–—―‣◦…§¶†‡‰′″〈〉"


def add_spaces_around_punctuation(text):
    # Add a space before and after all punctuation
    all_punctuation = string.punctuation + additional_punctuation
    return re.sub(r"([{}])".format(re.escape(all_punctuation)), r" \1 ", text)


class MultitaskTokenClassificationPipeline(Pipeline):

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "text" in kwargs:
            preprocess_kwargs["text"] = kwargs["text"]
        self.label_map = self.model.config.label_map
        self.id2label = {
            task: {id_: label for label, id_ in labels.items()}
            for task, labels in self.label_map.items()
        }
        return preprocess_kwargs, {}, {}

    # def preprocess(self, text, **kwargs):
    #
    #     language = detect(text)
    #     sentences = segment_and_trim_sentences(text, language, 512)
    #
    #     tokenized_inputs = self.tokenizer(
    #         text,
    #         padding="max_length",
    #         truncation=True,
    #         max_length=512,
    #         return_tensors="pt",
    #     )
    #
    #     text_sentences = [
    #         tokenize(add_spaces_around_punctuation(sentence)) for sentence in sentences
    #     ]
    #     return tokenized_inputs, text_sentences, text
    def preprocess(self, text, **kwargs):

        # sentences = segment_and_trim_sentences(text, language, 512)

        tokenized_inputs = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=512
        )

        text_sentence = tokenize(add_spaces_around_punctuation(text))
        return tokenized_inputs, text_sentence, text

    def _forward(self, inputs):
        inputs, text_sentences, text = inputs
        input_ids = torch.tensor([inputs["input_ids"]], dtype=torch.long).to(
            self.model.device
        )
        attention_mask = torch.tensor([inputs["attention_mask"]], dtype=torch.long).to(
            self.model.device
        )
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        return outputs, text_sentences, text

    # def _forward(self, inputs):
    #     inputs, text_sentences, text = inputs
    #     all_logits = {}
    #
    #     for i in range(len(text_sentences)):
    #         print(inputs["input_ids"][i].shape)
    #         input_ids = torch.tensor([inputs["input_ids"][i]], dtype=torch.long).to(
    #             self.model.device
    #         )
    #         attention_mask = torch.tensor(
    #             [inputs["attention_mask"][i]], dtype=torch.long
    #         ).to(self.model.device)
    #
    #         with torch.no_grad():
    #             outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    #
    #         # Accumulate logits for each task
    #         if not all_logits:
    #             all_logits = {task: logits for task, logits in outputs.logits.items()}
    #         else:
    #             for task in all_logits:
    #                 all_logits[task] = torch.cat(
    #                     (all_logits[task], outputs.logits[task]), dim=1
    #                 )
    #
    #     # Replace outputs.logits with accumulated logits
    #     outputs.logits = all_logits
    #
    #     return outputs, text_sentences, text

    def postprocess(self, outputs, **kwargs):
        """
        Postprocess the outputs of the model
        :param outputs:
        :param kwargs:
        :return:
        """
        tokens_result, text_sentence, text = outputs

        predictions = {}
        confidence_scores = {}
        for task, logits in tokens_result.logits.items():
            predictions[task] = torch.argmax(logits, dim=-1).tolist()
            confidence_scores[task] = F.softmax(logits, dim=-1).tolist()

        decoded_predictions = {}
        for task, preds in predictions.items():
            decoded_predictions[task] = [
                [self.id2label[task][label] for label in seq] for seq in preds
            ]
        entities = {}
        for task, preds in predictions.items():
            words_list, preds_list, confidence_list = realign(
                text_sentence,
                preds[0],
                confidence_scores[task][0],
                self.tokenizer,
                self.id2label[task],
            )

            entities[task] = get_entities(words_list, preds_list, confidence_list, text)

        return entities

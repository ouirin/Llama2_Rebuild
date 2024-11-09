import sys

import torch
from sentencepiece import SentencePieceProcessor

class Llama2Tokenizer:

    def __init__(self, vocab_file):

        self.vocab_file = vocab_file
        self.text_tokenizer = SentencePieceProcessor(str(vocab_file))
        self.n_words: int = self.text_tokenizer.vocab_size()

        self.bos_id: int = self.text_tokenizer.bos_id()
        self.eos_id: int = self.text_tokenizer.eos_id()
        self.pad_id: int = self.text_tokenizer.pad_id()

    def encode(self, text, bos=True, eos=False, truncation=None):

        tokens = self.text_tokenizer.encode(text)
        if truncation is not None:
            tokens = tokens[:2048]

        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]

        return tokens

    def decode(self, text_ids):

        text = self.text_tokenizer.decode(text_ids)

        return text

    def __call__(self, texts, padding=False, max_length=None, return_labels=False):

        input_ids = []

        for text in texts:
            input_ids.append(self.encode(text))

        attention_mask = []
        for input_id in input_ids:
            attention_mask.append([1] * len(input_id))

        for i in range(len(input_ids)):
            input_ids[i] = input_ids[i][:max_length]
            attention_mask[i] = attention_mask[i][:max_length]

        max_seq_length = max(map(lambda x: len(x), input_ids))

        if padding == "right":

            for i in range(len(input_ids)):
                pad_length = max_seq_length - len(input_ids[i])
                input_ids[i] = input_ids[i] + pad_length * [self.pad_id]
                attention_mask[i] = attention_mask[i] + pad_length * [0]

        if padding == "left" or padding is True:

            for i in range(len(input_ids)):
                pad_length = max_seq_length - len(input_ids[i])
                input_ids[i] = pad_length * [self.pad_id] + input_ids[i]
                attention_mask[i] = pad_length * [0] + attention_mask[i]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        if return_labels:
            labels = input_ids.masked_fill(~attention_mask.bool(), -100)
            return input_ids, labels

        return input_ids

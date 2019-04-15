import torch
import numpy as np

from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from .base import GenericEmbedder


class BertEmbedder(GenericEmbedder):
    def __init__(self, bert_model='bert-base-uncased', no_cuda=True) -> None:
        super().__init__()

        if bert_model not in ('bert-base-uncased', 'bert-large-uncased', 'bert-base-cased'):
            raise ValueError(f'Not supported BERT model: {bert_model}')

        self._hidden_units = 1024 if "large" in bert_model else 768

        do_lower_case = True if bert_model.endswith('uncased') else False
        self._tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

        self._device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        self._model = BertModel.from_pretrained(bert_model)
        self._model.to(self._device)

    def embed_sentence(self, text, tokenized=True, term_vectors=False, **kwargs):
        return next(iter(self.embed_collection([text], term_vectors=term_vectors, **kwargs)))

    def embed_collection(self, iterable, tokenized=True, term_vectors=False, **kwargs):
        layer_indexes = kwargs.get("layers", [-1, -2, -3, -4])
        seq_length = kwargs.get("max_seq_length", 128)
        batch_size = kwargs.get("batch_size", 32)

        all_tokens, all_input_ids, all_input_mask = [], [], []
        for tokens, input_ids, input_mask in self._convert_to_features(iterable, seq_length):
            all_tokens.append(tokens)
            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)

        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
        all_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        _data = TensorDataset(all_input_ids, all_input_mask, all_index)
        _sampler = SequentialSampler(_data)
        _dataloader = DataLoader(_data, sampler=_sampler, batch_size=batch_size)

        self._model.eval()

        for input_ids, input_mask, indices in _dataloader:
            input_ids = input_ids.to(self._device)
            input_mask = input_mask.to(self._device)

            all_encoder_layers, _ = self._model(input_ids, token_type_ids=None, attention_mask=input_mask)

            for b, idx in enumerate(indices):
                vectors = np.empty((0, self._hidden_units))

                for i in range(len(all_tokens[idx])):
                    for (j, layer_index) in enumerate(layer_indexes):
                        layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
                        layer_output = layer_output[b]

                        vectors = np.append(vectors, [layer_output[i]], axis=0)
                        # layers["values"] = [
                        #     round(x.item(), 6) for x in layer_output[i]
                        # ]
                        # all_layers.append(layers)
                vectors = vectors.reshape((len(layer_indexes), len(all_tokens[idx]), -1))
                yield self._get_term_or_seq_vectors(vectors, term_vectors, kwargs.get("pooling", "concat"))

    def _convert_to_features(self, iterable, seq_length):
        for text in iterable:
            tokens = ["[CLS]"]
            for token in self._tokenizer.tokenize(text):
                tokens.append(token)
            tokens.append("[SEP]")
            input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            if len(tokens) > seq_length - 2:
                tokens = tokens[0:(seq_length - 2)]

            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)

            yield tokens, input_ids, input_mask

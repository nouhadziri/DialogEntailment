import numpy as np

from ..util.nlp import Spacy


class GenericEmbedder:
    def __init__(self) -> None:
        self._spacy = Spacy()

    def _get_term_or_seq_vectors(self, vectors, term_vectors, pooling_method):
        pooled_embeddings = pool_layers(vectors, pooling_method)
        if term_vectors:
            return pooled_embeddings
        else:
            return np.mean(pooled_embeddings, axis=0)

    def embed_sentence(self, text, tokenized=True, term_vectors=False, **kwargs):
        raise NotImplementedError()

    def embed_collection(self, iterable, tokenized=True, term_vectors=False, **kwargs):
        raise NotImplementedError()


def pool_layers(vectors, pooling_method):
    if pooling_method == "concat":
        return vectors.transpose(1, 0, 2).reshape(vectors.shape[1], -1)
    elif pooling_method == "mean":
        return np.mean(vectors, axis=0)
    elif pooling_method == "max":
        return np.amax(vectors, axis=0)


class TokenizerWrapper:
    def __init__(self, spacy, iterable, tokenized=True) -> None:
        self._spacy = spacy
        self._iterable = iterable
        self._tokenized = tokenized

    def __iter__(self):
        for text in self._iterable:
            if self._tokenized:
                yield text.split()
            else:
                yield self._spacy.word_tokenize(text)

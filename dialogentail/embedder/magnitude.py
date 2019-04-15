
from pymagnitude import *
from .base import GenericEmbedder, TokenizerWrapper


class MagnitudeEmbedder(GenericEmbedder):
    def __init__(self, magnitude_model) -> None:
        super().__init__()

        supported_models = ('word2vec/medium/GoogleNews-vectors-negative300.magnitude',
                            'glove/medium/glove.840B.300d.magnitude',
                            'glove/medium/glove.6B.300d.magnitude',
                            'glove/medium/glove.6B.50d.magnitude',
                            'fasttext/medium/crawl-300d-2M.magnitude',
                            'fasttext/medium/wiki-news-300d-1M-subword.magnitude')

        if magnitude_model not in supported_models:
            raise ValueError(f'Not supported Magnitude model: {magnitude_model}')

        self._vectors = Magnitude(MagnitudeUtils.download_model(magnitude_model))

    def _get_term_or_seq_vectors(self, vectors, term_vectors, pooling_method):
        if term_vectors:
            return vectors
        else:
            return aggregate_vectors(vectors, pooling_method)

    def embed_sentence(self, text, tokenized=True, term_vectors=False, **kwargs):
        if tokenized:
            tokens = text.split()
        else:
            tokens = self._spacy.word_tokenize(text)

        embeddings = self._vectors.query(tokens)
        return self._get_term_or_seq_vectors(embeddings, term_vectors, kwargs.get("pooling", "mean"))

    def embed_collection(self, iterable, tokenized=True, term_vectors=False, **kwargs):
        for tokens in TokenizerWrapper(self._spacy, iterable, tokenized):
            embeddings = self._vectors.query(tokens)
            yield self._get_term_or_seq_vectors(embeddings, term_vectors, kwargs.get("pooling", "mean"))


def aggregate_vectors(vectors, aggregate_method):
    if aggregate_method == "mean":
        return np.mean(vectors, axis=0)
    elif aggregate_method == "max":
        return np.amax(vectors, axis=0)


class GloVeEmbedder(MagnitudeEmbedder):
    def __init__(self, model_name='840B.300d') -> None:

        if model_name not in ('840B.300d', '6B.300d', '6B.50d'):
            raise ValueError(f'Not supported GloVe model')

        super().__init__(f'glove/medium/glove.{model_name}.magnitude')


class Word2VecEmbedder(MagnitudeEmbedder):
    def __init__(self) -> None:
        super().__init__(f'word2vec/medium/GoogleNews-vectors-negative300.magnitude')

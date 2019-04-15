from allennlp.commands.elmo import ElmoEmbedder as AllenNLPElmoEmbedder

from .base import GenericEmbedder, TokenizerWrapper


class ElmoEmbedder(GenericEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self._elmo = AllenNLPElmoEmbedder()

    def embed_sentence(self, text, tokenized=True, term_vectors=False, **kwargs):
        if tokenized:
            tokens = text.split()
        else:
            tokens = self._spacy.word_tokenize(text)

        vectors = self._elmo.embed_sentence(tokens)
        return self._get_term_or_seq_vectors(vectors, term_vectors, kwargs.get("pooling", "mean"))

    def embed_collection(self, iterable, tokenized=True, term_vectors=False, **kwargs):
        collection_vectors = self._elmo.embed_sentences(TokenizerWrapper(self._spacy, iterable, tokenized))
        for vectors in collection_vectors:
            yield self._get_term_or_seq_vectors(vectors, term_vectors, kwargs.get("pooling", "mean"))

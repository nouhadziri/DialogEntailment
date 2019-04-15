
from .bert import BertEmbedder
from .elmo import ElmoEmbedder
from .magnitude import GloVeEmbedder, Word2VecEmbedder


def from_embedding(embedding_method):
    if embedding_method.lower().startswith("bert"):
        if embedding_method.lower() == "bert":
            return BertEmbedder()
        else:
            return BertEmbedder(embedding_method)
    elif embedding_method.lower() == "elmo":
        return ElmoEmbedder()
    elif embedding_method.lower() == "glove":
        return GloVeEmbedder()
    elif embedding_method.lower() == "word2vec":
        return Word2VecEmbedder()

    raise ValueError(f"Unsupported embedding method: {embedding_method}")

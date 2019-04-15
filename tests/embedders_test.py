import unittest

from dialogentail.embedder.elmo import ElmoEmbedder
from dialogentail.embedder.bert import BertEmbedder
from dialogentail.embedder.magnitude import GloVeEmbedder


def skip_class(base_cls):
    class BaseClassSkipper(base_cls):
        @classmethod
        def setUpClass(cls):
            if cls is BaseClassSkipper:
                raise unittest.SkipTest("Base class")
            super().setUpClass()
    return BaseClassSkipper


@skip_class
class EmbedderTest(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

        self.test_sentence = 'I used to eat apple when working at Apple'
        self.test_sentences = [
            "I ate an apple for breakfast.",
            "I ate a carrot for breakfast."
        ]


@skip_class
class ElmoEmbedderTest(EmbedderTest):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        # self._embedder = ElmoEmbedder()

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._embedder = ElmoEmbedder()

    def test_embed_sentence(self):
        sent_vector = self._embedder.embed_sentence(self.test_sentence)
        self.assertEqual(sent_vector.shape, (1024,))

        term_vectors = self._embedder.embed_sentence(self.test_sentence, term_vectors=True)
        self.assertEqual(term_vectors.shape, (len(self.test_sentence.split()), 1024))

    def test_embed_iterable(self):
        vectors = self._embedder.embed_collection(self.test_sentences, tokenized=False)

        i = 0
        for vec in vectors:
            self.assertEqual(vec.shape, (1024,))
            i += 1

        self.assertEqual(i, len(self.test_sentences))


@skip_class
class BertEmbedderTest(EmbedderTest):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._embedder = BertEmbedder('bert-base-cased')

    def test_embed_sentence(self):
        sent_vector = self._embedder.embed_sentence(self.test_sentence)
        self.assertEqual(sent_vector.shape, (4*768,))

        term_vectors = self._embedder.embed_sentence(self.test_sentence, term_vectors=True)
        # Two special tokens (i.e., [CLS] and [SEP]) will be added to the text
        self.assertEqual(term_vectors.shape, (len(self.test_sentence.split()) + 2, 4*768))

    def test_embed_iterable(self):
        vectors = self._embedder.embed_collection(self.test_sentences, pooling='mean')

        i = 0
        for vec in vectors:
            self.assertEqual(vec.shape, (768,))
            i += 1

        self.assertEqual(i, len(self.test_sentences))


class GloVeEmbedderTest(EmbedderTest):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._embedder = GloVeEmbedder('6B.50d')

    def test_embed_sentence(self):
        sent_vector = self._embedder.embed_sentence(self.test_sentence)
        self.assertEqual(sent_vector.shape, (50,))

        term_vectors = self._embedder.embed_sentence(self.test_sentence, term_vectors=True)
        self.assertEqual(term_vectors.shape, (len(self.test_sentence.split()), 50))

    def test_embed_iterable(self):
        vectors = self._embedder.embed_collection(self.test_sentences, tokenized=False)

        i = 0
        for vec in vectors:
            self.assertEqual(vec.shape, (50,))
            i += 1

        self.assertEqual(i, len(self.test_sentences))

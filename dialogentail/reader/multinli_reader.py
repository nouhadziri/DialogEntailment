import codecs

from ..util.nlp import Spacy


class MultiNLIReader:
    def __init__(self, mnli_path, **kwargs):
        self._mnli_path = mnli_path
        self._tokenized = kwargs.get("tokenized", False)
        self._do_lowercase = kwargs.get("do_lowercase", True)
        if not self._tokenized:
            self._spacy = Spacy()

    def _normalize(self, text):
        normalized_text = " ".join(self._spacy.word_tokenize(text.replace('I\\m', "I'm"))).strip()
        if self._do_lowercase:
            normalized_text = normalized_text.lower()

        return normalized_text

    def __iter__(self):
        with codecs.getreader("utf-8")(open(self._mnli_path, "rb")) as mnli_file:
            for i, line in enumerate(mnli_file):
                if i == 0:
                    continue

                line = line.strip()
                if not line:
                    continue

                tokens = line.split("\t")
                gold_label = tokens[0]
                s1, s2 = self._normalize(tokens[5]), self._normalize(tokens[6])
                promptId, pairId = tokens[7], tokens[8]
                genre = tokens[9]

                yield s1, s2, promptId, pairId, genre, gold_label

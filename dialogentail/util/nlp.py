from string import punctuation
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


def omit_punctuations(txt):
    return list(set([tok for tok in txt.split() if tok not in punctuation]))


def omit_stopwords(tokens):
    return [w for w in tokens if not is_stopword_or_punct(w)]


def is_stopword_or_punct(word):
    return word in STOP_WORDS or word in punctuation


class Spacy:
    def __init__(self, enable_ner=False, enable_tagger=False, enable_parser=False):
        components_to_exclude = []
        if not enable_ner:
            components_to_exclude.append('ner')
        if not enable_tagger:
            components_to_exclude.append('tagger')
        if not enable_parser:
            components_to_exclude.append('parser')

        self._nlp = spacy.load('en', disable=components_to_exclude)

    def word_tokenize(self, text):
        doc = self._nlp(text)
        return [token.text for token in doc]

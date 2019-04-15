import math
from scipy.spatial import distance

from .semantic_similarity import SemanticSimilarity, _find_interesting_segments
from .util import nlp


class SemanticDistance(SemanticSimilarity):
    """
    The output of this metric is reported in the paper.
    """
    def __init__(self, embedding_method='elmo', separator='[SEP]', boost_factor=True, dull_responses_file=None):
        super().__init__(embedding_method, separator, boost_factor, dull_responses_file)

    def _calc_similarity(self, response, v_response, v_ref):
        if self._boost_factor:
            # number of non-stop words
            ns = len(nlp.omit_stopwords(response.split()))

            # number of words not in dull response pattern
            m = len(nlp.omit_stopwords(_find_interesting_segments(response, self._dull_responses)))
            coef = 1.0 + math.log10((2.0 + ns) / (2.0 + m))
        else:
            coef = 1.0

        return distance.cosine(v_ref, v_response) * coef

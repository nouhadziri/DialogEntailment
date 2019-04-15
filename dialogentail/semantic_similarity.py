import math
import re
import os
from pathlib import Path

from scipy.spatial import distance

from .util import nlp, files
from .eval_metric import EvaluationMetric
from .embedder.factory import from_embedding
from .reader.response_reader import ResponseFileReader


class SemanticSimilarity(EvaluationMetric):
    def __init__(self, embedding_method='elmo', separator='[SEP]', boost_factor=True, dull_responses_file=None):
        self._embedder = from_embedding(embedding_method)
        self._embedding_method = embedding_method
        self._separator = separator
        self._boost_factor = boost_factor
        self._dull_responses = _load_dull_responses(dull_responses_file)

    def compute_metric(self, conversation_history, actual_response, generated_response, **kwargs):
        collection = []
        collection.append(self._separator.join(conversation_history))
        for utterance in conversation_history:
            collection.append(utterance)
        collection.append(actual_response)
        collection.append(generated_response)

        vectors = [vec for vec in self._embedder.embed_collection(collection, **kwargs)]

        gt_result = {
            "i": 0,
            "SS_context": self._calc_similarity(actual_response, vectors[-2], vectors[0]),
            "delta_SS_context": 0.0,
            "gen_type": "ground_truth"
        }

        gen_result = {
            "i": 0,
            "SS_context": self._calc_similarity(generated_response, vectors[-1], vectors[0]),
            "gen_type": "unknown"
        }
        gen_result["delta_SS_context"] = gen_result["SS_context"] - gt_result["SS_context"]

        for i in range(len(conversation_history)):
            end_i = i - len(conversation_history)
            gt_result[f"SS_Utt_{end_i}"] = self._calc_similarity(actual_response, vectors[-2], vectors[i + 1])
            gt_result[f"delta_SS_Utt_{end_i}"] = 0.0
            gen_result[f"SS_Utt_{end_i}"] = self._calc_similarity(generated_response, vectors[-1], vectors[i + 1])
            gen_result[f"delta_SS_Utt_{end_i}"] = gen_result[f"SS_Utt_{end_i}"] - gt_result[f"SS_Utt_{end_i}"]

        return [gt_result, gen_result]

    def _calc_similarity(self, response, v_response, v_ref):
        if self._boost_factor:
            # number of non-stop words
            ns = len(nlp.omit_stopwords(response.split()))

            # number of words not in dull response pattern
            m = len(nlp.omit_stopwords(_find_interesting_segments(response, self._dull_responses)))
            coef = 1.0 + math.log10((2.0 + m) / (2.0 + ns))
        else:
            coef = 1.0

        return (1 - distance.cosine(v_ref, v_response)) * coef

    def compute_metric_for_file(self, response_file, generator_methods, cache_dir=None):
        if cache_dir is None:
            cache_dir = Path.home() / '.dialog_eval'
            cache_dir.mkdir(exist_ok=True)
        else:
            cache_dir = Path(cache_dir)

        embedding_cache = cache_dir / f"{files.get_file_name(response_file)}.{self._embedding_method}.pkl"

        n_generators = len(generator_methods)
        if not embedding_cache.exists():
            vectors = self._cache_embeddings(embedding_cache, response_file, n_generators)
        else:
            vectors = files.load_obj(embedding_cache)

        result = []

        i = 0
        for j, (conversation_history, actual_response, generated_responses) in \
                enumerate(ResponseFileReader(response_file, n_generators)):
            v_context = vectors[i]
            i += 1

            v_utterances = []
            for _ in range(len(conversation_history)):
                v_utterances.append(vectors[i])
                i += 1

            v_actual = vectors[i]
            i += 1
            gt_result = {
                "i": j,
                "SS_context": self._calc_similarity(actual_response, v_actual, v_context),
                "delta_SS_context": 0.0,
                "gen_type": "ground_truth"
            }

            for u, v_utt in enumerate(v_utterances):
                gt_result[f"SS_Utt_{u - len(v_utterances)}"] = self._calc_similarity(actual_response,
                                                                                     v_actual,
                                                                                     v_utt)
                gt_result[f"delta_SS_Utt_{u - len(v_utterances)}"] = 0.0

            result.append(gt_result)

            for generated_response, method in zip(generated_responses, generator_methods):
                v_resp = vectors[i]
                i += 1

                gen_result = {
                    "i": j,
                    "SS_context": self._calc_similarity(generated_response, v_resp, v_context),
                    "gen_type": method
                }
                gen_result["delta_SS_context"] = gen_result["SS_context"] - gt_result["SS_context"]

                for u, v_utt in enumerate(v_utterances):
                    end_i = u - len(v_utterances)
                    gen_result[f"SS_Utt_{end_i}"] = self._calc_similarity(generated_response,
                                                                          v_resp,
                                                                          v_utt)
                    gen_result[f"delta_SS_Utt_{end_i}"] = gen_result[f"SS_Utt_{end_i}"] - \
                                                          gt_result[f"SS_Utt_{end_i}"]

                result.append(gen_result)

        return result

    def _text_iterator(self, response_file, n_generator_methods):
        for conversation_history, actual_response, generated_responses in \
                ResponseFileReader(response_file, n_generator_methods):
            yield self._separator.join(conversation_history)
            for utterance in conversation_history:
                yield utterance

            yield actual_response

            for generated_response in generated_responses:
                yield generated_response

    def _cache_embeddings(self, cache_file, response_file, n_generator_methods):
        vectors = []
        for vec in self._embedder.embed_collection(self._text_iterator(response_file, n_generator_methods)):
            vectors.append(vec)

        files.save_obj(vectors, cache_file)
        return vectors


def _load_dull_responses(dull_response_file=None):
    if dull_response_file is None:
        basedir = files.get_containing_dir(__file__)
        dull_response_file = os.path.join(basedir, 'dull_responses.txt')

    dull_responses = []
    with open(dull_response_file, 'r') as reader:
        for sample_dull_resp in reader:
            dull_responses.append('^' + sample_dull_resp.strip().replace('.', '\.').replace('$1', '(.+)') + '$')

    return dull_responses


def _find_interesting_segments(response, dull_responses):
    matched_segment = []

    has_matched = False
    for dr in dull_responses:
        matched = re.match(dr, response)
        if not matched:
            continue

        has_matched = True
        segment = [w for g in matched.groups() for w in g.split()]
        if len(segment) < len(matched_segment) or not matched_segment:
            matched_segment = segment

    return matched_segment if has_matched else response.split()

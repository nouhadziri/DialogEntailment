import logging
from typing import List, Dict, Any, Generator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from allennlp.predictors.predictor import Predictor
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, WEIGHTS_NAME

from .eval_metric import EvaluationMetric
from .reader.response_reader import ResponseFileReader
from .huggingface.finetune_bert import InputExample, convert_examples_to_features

logger = logging.getLogger(__name__)


class Coherence(EvaluationMetric):
    def __init__(self, model_path: str, separator=" "):
        self._model_path = model_path
        self._separator = separator

    def _create_instances_from_file(self, response_file: str, n_generator_methods: int) -> Generator[InputExample, None, None]:
        for i, (conversation_history, actual_response, generated_responses) in \
                enumerate(ResponseFileReader(response_file, n_generator_methods)):
            j = 0
            context_premise = self._separator.join(conversation_history)
            yield InputExample(f"test-{i}-{j}", context_premise, actual_response)
            j += 1

            for utterance in conversation_history:
                yield InputExample(f"test-{i}-{j}", utterance, actual_response)
                j += 1

            for generated_response in generated_responses:
                yield InputExample(f"test-{i}-{j}", context_premise, generated_response)
                j += 1

                for utterance in conversation_history:
                    yield InputExample(f"test-{i}-{j}", utterance, generated_response)
                    j += 1

    def _create_instances_from_input(self, conversation_history: List[str],
                                     actual_response: str,
                                     generated_response: str) -> Generator[InputExample, None, None]:
        context_premise = self._separator.join(conversation_history)
        i = 1
        yield InputExample(f"test-{i}", context_premise, actual_response)
        i += 1

        yield InputExample(f"test-{i}", context_premise, generated_response)
        i += 1

        for utterance in conversation_history:
            yield InputExample(f"test-{i}", utterance, actual_response)
            i += 1
            yield InputExample(f"test-{i}", utterance, generated_response)
            i += 1


class ESIMCoherence(Coherence):
    def __init__(self, model_archive: str):
        super().__init__(model_archive)
        self._predictor = Predictor.from_path(self._model_path, "textual-entailment")
        self._labels = ["entailment", "contradiction", "neutral"]

    def compute_metric(self, conversation_history: List[str], actual_response: str, generated_response: str) -> \
            List[Dict[str, Any]]:
        test_data = self._create_json_dicts(
            self._create_instances_from_input(conversation_history, actual_response, generated_response))
        predictions = self._predictor.predict_batch_json(test_data)

        gt_result = {
            "i": 0,
            "context_label": self._labels[np.argmax(predictions[0]["label_probs"])],
            "gen_type": "ground_truth"
        }

        gen_result = {
            "i": 0,
            "context_label": self._labels[np.argmax(predictions[1]["label_probs"])],
            "gen_type": "unknown"
        }

        for i in range(2, len(predictions), 2):
            rev_i = i / 2 - len(conversation_history)
            gt_result[f"Utt_{rev_i}_label"] = self._labels[np.argmax(predictions[i]["label_probs"])]
            gen_result[f"Utt_{rev_i}_label"] = self._labels[np.argmax(predictions[i + 1]["label_probs"])]

        return [gt_result, gen_result]

    def _create_json_dicts(self, input_examples: Generator[InputExample, None, None]) -> List[Dict[str, str]]:
        return [
            {u"premise": example.text_a, u"hypothesis": example.text_b} for example in input_examples
        ]

    def compute_metric_for_file(self, response_file: str, generator_methods: List[str]) -> List[Dict[str, Any]]:
        n_generators = len(generator_methods)
        test_data = self._create_json_dicts(self._create_instances_from_file(response_file, n_generators))
        predictions = self._predictor.predict_batch_json(test_data)

        result = []
        i = 0
        for j, (conversation_history, actual_response, generated_responses) in \
                enumerate(ResponseFileReader(response_file, n_generators)):

            gt_result = {
                "i": j,
                "context_label": self._labels[np.argmax(predictions[i]["label_probs"])],
                "gen_type": "ground_truth"
            }
            i += 1

            for u in range(len(conversation_history)):
                rev_u = u - len(conversation_history)
                gt_result[f"Utt_{rev_u}_label"] = self._labels[np.argmax(predictions[i]["label_probs"])]
                i += 1
            result.append(gt_result)

            for generated_response, method in zip(generated_responses, generator_methods):
                gen_result = {
                    "i": j,
                    "context_label": self._labels[np.argmax(predictions[i]["label_probs"])],
                    "gen_type": method
                }
                i += 1

                for u in range(len(conversation_history)):
                    rev_u = u - len(conversation_history)
                    gen_result[f"Utt_{rev_u}_label"] = self._labels[np.argmax(predictions[i]["label_probs"])]
                    i += 1

                result.append(gen_result)

        return result


class BertCoherence(Coherence):
    def __init__(self, model_dir: str, separator=" ", bert_model='bert-base-uncased', max_seq_length=128,
                 batch_size=16):
        super().__init__(model_dir, separator)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._n_gpu = torch.cuda.device_count()

        self._tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=bert_model.endswith('uncased'))

        output_model_file = Path(self._model_path) / WEIGHTS_NAME
        model_state_dict = torch.load(output_model_file)
        self._model = BertForSequenceClassification.from_pretrained(bert_model, num_labels=3,
                                                                    state_dict=model_state_dict)
        self._model.to(self._device)
        if self._n_gpu > 1:
            self._model = torch.nn.DataParallel(self._model)

        self._max_seq_length = max_seq_length
        self._batch_size = batch_size
        self._labels = ["contradiction", "entailment", "neutral"]

    def _eval_model(self, eval_examples):
        eval_features = convert_examples_to_features(
            eval_examples, self._labels, self._max_seq_length, self._tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self._batch_size)

        self._model.eval()

        all_predicted_labels = np.empty((0,), dtype=np.int8)
        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
            input_ids = input_ids.to(self._device)
            input_mask = input_mask.to(self._device)
            segment_ids = segment_ids.to(self._device)

            with torch.no_grad():
                logits = self._model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            predicted_labels = np.argmax(logits, axis=1)
            all_predicted_labels = np.append(all_predicted_labels, predicted_labels, axis=0)

        return all_predicted_labels

    def compute_metric(self, conversation_history: List[str], actual_response: str, generated_response: str) \
            -> List[Dict[str, Any]]:
        test_generator = self._create_instances_from_input(conversation_history, actual_response, generated_response)
        predictions = self._eval_model(test_generator)

        gt_result = {
            "i": 0,
            "context_label": self._labels[predictions[0]],
            "gen_type": "ground_truth"
        }

        gen_result = {
            "i": 0,
            "context_label": self._labels[predictions[1]],
            "gen_type": "unknown"
        }

        for i in range(2, len(predictions), 2):
            rev_i = i / 2 - len(conversation_history)
            gt_result[f"Utt_{rev_i}_label"] = self._labels[predictions[i]]
            gen_result[f"Utt_{rev_i}_label"] = self._labels[predictions[i + 1]]

        return [gt_result, gen_result]

    def compute_metric_for_file(self, response_file: str, generator_methods: List[str]) -> List[Dict[str, Any]]:
        n_generators = len(generator_methods)
        test_generator = self._create_instances_from_file(response_file, n_generators)
        predictions = self._eval_model(test_generator)

        result = []
        i = 0
        for j, (conversation_history, actual_response, generated_responses) in \
                enumerate(ResponseFileReader(response_file, n_generators)):

            gt_result = {
                "i": j,
                "context_label": self._labels[predictions[i]],
                "gen_type": "ground_truth"
            }
            i += 1

            for u in range(len(conversation_history)):
                rev_u = u - len(conversation_history)
                gt_result[f"Utt_{rev_u}_label"] = self._labels[predictions[i]]
                i += 1
            result.append(gt_result)

            for generated_response, method in zip(generated_responses, generator_methods):
                gen_result = {
                    "i": j,
                    "context_label": self._labels[predictions[i]],
                    "gen_type": method
                }
                i += 1

                for u in range(len(conversation_history)):
                    rev_u = u - len(conversation_history)
                    gen_result[f"Utt_{rev_u}_label"] = self._labels[predictions[i]]
                    i += 1

                result.append(gen_result)

        return result

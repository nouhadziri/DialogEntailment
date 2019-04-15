from typing import List, Dict, Any


class EvaluationMetric:
    def compute_metric(self, conversation_history: str, actual_response: str, generated_response: str) -> List[Dict[str, Any]]:
        raise NotImplementedError()

    def compute_metric_for_file(self, response_file: str, generator_methods: List[str]) -> List[Dict[str, Any]]:
        raise NotImplementedError()

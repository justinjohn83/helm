from __future__ import annotations

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric, MetricMetadata
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class CodeSetMetric(Metric):
    """Evaluate comma-separated code list generations with set-based metrics."""

    def __init__(self, delimiter: str = ", "):
        self.delimiter = delimiter

    @staticmethod
    def _parse_codes(text: str) -> set[str]:
        if not text:
            return set()
        return {token.strip().upper() for token in text.split(",") if token.strip()}

    @staticmethod
    def _safe_divide(numerator: int, denominator: int) -> float:
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        del adapter_spec, metric_service, eval_cache_path

        reference = request_state.instance.first_correct_reference
        if reference is None:
            return []

        predicted_text = ""
        if request_state.result and request_state.result.completions:
            predicted_text = request_state.result.completions[0].text

        gold_set = self._parse_codes(reference.output.text)
        predicted_set = self._parse_codes(predicted_text)

        overlap = len(gold_set & predicted_set)
        precision = self._safe_divide(overlap, len(predicted_set))
        recall = self._safe_divide(overlap, len(gold_set))
        iou = self._safe_divide(overlap, len(gold_set | predicted_set))
        f1_denominator = len(gold_set) + len(predicted_set)
        f1 = self._safe_divide(2 * overlap, f1_denominator)
        exact = 1.0 if gold_set == predicted_set else 0.0

        return [
            Stat(MetricName("code_set_f1")).add(f1),
            Stat(MetricName("code_set_precision")).add(precision),
            Stat(MetricName("code_set_recall")).add(recall),
            Stat(MetricName("code_set_iou")).add(iou),
            Stat(MetricName("code_set_exact")).add(exact),
        ]

    def get_metadata(self) -> List[MetricMetadata]:
        return [
            MetricMetadata(name="code_set_f1", lower_is_better=False),
            MetricMetadata(name="code_set_precision", lower_is_better=False),
            MetricMetadata(name="code_set_recall", lower_is_better=False),
            MetricMetadata(name="code_set_iou", lower_is_better=False),
            MetricMetadata(name="code_set_exact", lower_is_better=False),
        ]

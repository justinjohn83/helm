from __future__ import annotations

from types import SimpleNamespace

from helm.benchmark.metrics.medhelm.code_set_metrics import CodeSetMetric
from helm.benchmark.scenarios.scenario import CORRECT_TAG, Input, Instance, Output, Reference


def _build_request_state(gold: str | None, prediction: str) -> SimpleNamespace:
    references = []
    if gold is not None:
        references.append(Reference(output=Output(text=gold), tags=[CORRECT_TAG]))
    instance = Instance(input=Input(text="prompt"), references=references)
    result = SimpleNamespace(completions=[SimpleNamespace(text=prediction)])
    return SimpleNamespace(instance=instance, result=result)


def _stats_by_name(stats: list) -> dict[str, float]:
    return {stat.name.name: float(stat.mean or 0.0) for stat in stats}


def test_code_set_metric_perfect_match_order_insensitive() -> None:
    metric = CodeSetMetric()
    request_state = _build_request_state("0213T, 0216T, 12001", "12001, 0213T, 0216T")

    values = _stats_by_name(metric.evaluate_generation(None, request_state, None, ""))

    assert values["code_set_f1"] == 1.0
    assert values["code_set_precision"] == 1.0
    assert values["code_set_recall"] == 1.0
    assert values["code_set_iou"] == 1.0
    assert values["code_set_exact"] == 1.0


def test_code_set_metric_partial_overlap() -> None:
    metric = CodeSetMetric()
    request_state = _build_request_state("0213T, 0216T, 12001, 12002", "0213T, 0216T, 99999")

    values = _stats_by_name(metric.evaluate_generation(None, request_state, None, ""))

    assert round(values["code_set_f1"], 6) == round(4 / 7, 6)
    assert round(values["code_set_precision"], 6) == round(2 / 3, 6)
    assert values["code_set_recall"] == 0.5
    assert values["code_set_iou"] == 0.4
    assert values["code_set_exact"] == 0.0


def test_code_set_metric_handles_empty_prediction_and_normalization() -> None:
    metric = CodeSetMetric()
    request_state = _build_request_state("0213t , 0216T,12001", "")

    values = _stats_by_name(metric.evaluate_generation(None, request_state, None, ""))

    assert values["code_set_f1"] == 0.0
    assert values["code_set_precision"] == 0.0
    assert values["code_set_recall"] == 0.0
    assert values["code_set_iou"] == 0.0
    assert values["code_set_exact"] == 0.0


def test_code_set_metric_handles_single_code_and_trailing_comma() -> None:
    metric = CodeSetMetric()
    request_state = _build_request_state("47562", "47562,")

    values = _stats_by_name(metric.evaluate_generation(None, request_state, None, ""))

    assert values["code_set_f1"] == 1.0
    assert values["code_set_precision"] == 1.0
    assert values["code_set_recall"] == 1.0
    assert values["code_set_iou"] == 1.0
    assert values["code_set_exact"] == 1.0


def test_code_set_metric_returns_no_stats_without_correct_reference() -> None:
    metric = CodeSetMetric()
    request_state = _build_request_state(None, "0213T")

    assert metric.evaluate_generation(None, request_state, None, "") == []

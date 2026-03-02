from __future__ import annotations

import pytest

from helm.benchmark.run_specs.medhelm.benchmark_config import BenchmarkConfig, SimpleMetricConfig


def _build_config(metrics: list[SimpleMetricConfig]) -> BenchmarkConfig:
    return BenchmarkConfig(
        name="test_benchmark",
        description="test",
        prompt_file="prompt.txt",
        dataset_file="dataset.csv",
        main_metric=metrics[0],
        metrics=metrics,
        max_tokens=64,
    )


def test_get_metric_specs_maps_code_set_f1() -> None:
    benchmark_config = _build_config(
        [SimpleMetricConfig(name="exact_match"), SimpleMetricConfig(name="code_set_f1")]
    )

    metric_specs = benchmark_config.get_metric_specs()
    class_names = {metric_spec.class_name for metric_spec in metric_specs}
    code_set_metric = [
        metric_spec for metric_spec in metric_specs if metric_spec.class_name.endswith("CodeSetMetric")
    ]

    assert "helm.benchmark.metrics.medhelm.code_set_metrics.CodeSetMetric" in class_names
    assert len(code_set_metric) == 1
    assert code_set_metric[0].args == {"delimiter": ", "}


def test_get_metric_specs_unknown_metric_raises_value_error() -> None:
    benchmark_config = _build_config([SimpleMetricConfig(name="unknown_metric")])

    with pytest.raises(ValueError, match="Unknown metric name"):
        benchmark_config.get_metric_specs()

from __future__ import annotations

from pathlib import Path

from helm.benchmark.scenarios.medhelm_configurable_scenario import MedHELMConfigurableScenario


def _write_files(tmp_path: Path, dataset_contents: str) -> Path:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Question: {question}", encoding="utf-8")

    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text(dataset_contents, encoding="utf-8")

    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test_medhelm",
                "description: test benchmark",
                f"prompt_file: {prompt_path}",
                f"dataset_file: {dataset_path}",
                "metrics:",
                "  - name: exact_match",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_get_instances_sets_sub_split_from_category(tmp_path: Path) -> None:
    config_path = _write_files(
        tmp_path,
        "question,correct_answer,category\nIs CPT A bundled with B?,Yes,bundling_binary\n",
    )
    scenario = MedHELMConfigurableScenario(name="test_medhelm", config_path=str(config_path))

    instances = scenario.get_instances(output_path=str(tmp_path))

    assert len(instances) == 1
    assert instances[0].sub_split == "bundling_binary"


def test_get_instances_falls_back_to_none_without_category(tmp_path: Path) -> None:
    config_path = _write_files(
        tmp_path,
        "question,correct_answer\nWhat code is used for X?,12345\n",
    )
    scenario = MedHELMConfigurableScenario(name="test_medhelm", config_path=str(config_path))

    instances = scenario.get_instances(output_path=str(tmp_path))

    assert len(instances) == 1
    assert instances[0].sub_split is None

from pathlib import Path

from carsguard.core.config import deep_update, load_project_configs


def test_deep_update():
    base = {"a": 1, "b": {"x": 10, "y": 20}}
    updates = {"b": {"y": 99, "z": 5}, "c": 3}

    merged = deep_update(base, updates)

    assert merged["a"] == 1
    assert merged["b"]["x"] == 10
    assert merged["b"]["y"] == 99
    assert merged["b"]["z"] == 5
    assert merged["c"] == 3


def test_load_project_configs():
    cfg = load_project_configs(Path("configs"))

    assert "paths" in cfg
    assert "preprocessing" in cfg
    assert "references" in cfg
    assert "scoring" in cfg

    assert "benchmark_table" in cfg["paths"]
    assert "features" in cfg["scoring"]
    assert "artifact_detection" in cfg["scoring"]
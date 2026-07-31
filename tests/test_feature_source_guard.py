from pathlib import Path


def test_feature_module_contains_no_known_future_leakage_operations() -> None:
    source = Path("code/gas_forecast/features.py").read_text(encoding="utf-8")
    forbidden = ("shift(-", "center=True", ".bfill(", "backfill", "limit_direction=\"both\"")
    assert not [token for token in forbidden if token in source]


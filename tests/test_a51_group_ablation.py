from __future__ import annotations

from scripts.run_a51_group_ablation import _status


def _comparison(*, pooled_difference: float, recent: dict[str, float]) -> dict[str, object]:
    return {
        "pooled_difference": pooled_difference,
        "recent_5_folds_difference": recent,
    }


def test_a56_retains_only_fixed_stability_contract() -> None:
    rich_gas = _comparison(
        pooled_difference=-0.000051,
        recent={"dev_15": -0.0001, "dev_16": 0.0001, "dev_17": -0.0001, "dev_18": -0.0001, "dev_19": 0.0001},
    )
    a51 = _comparison(pooled_difference=0.000009, recent={})

    result = _status(rich_gas, a51)

    assert result["status"] == "RETAIN_STABILITY"
    assert result["recent5_wins_vs_rich_gas"] == 3


def test_a56_does_not_trade_away_a51_gain_for_recent_wins() -> None:
    rich_gas = _comparison(
        pooled_difference=-0.00006,
        recent={"dev_15": -0.0001, "dev_16": -0.0001, "dev_17": -0.0001, "dev_18": 0.0001, "dev_19": 0.0001},
    )
    a51 = _comparison(pooled_difference=0.000011, recent={})

    result = _status(rich_gas, a51)

    assert result["status"] == "DO_NOT_RETAIN"

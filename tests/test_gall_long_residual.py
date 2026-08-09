from __future__ import annotations

from scripts.run_gall_long_residual import _status


def _comparison(
    *,
    pooled_difference: float,
    generator_all_difference: float,
    recent: dict[str, float],
) -> dict[str, object]:
    return {
        "pooled_difference": pooled_difference,
        "recent_5_folds_difference": recent,
        "pairwise": {
            "by_target": {
                "generator_all": {"difference": generator_all_difference},
            }
        },
    }


def test_a60_retains_only_when_pooled_target_and_recent_conditions_hold() -> None:
    result = _status(
        _comparison(
            pooled_difference=-0.000051,
            generator_all_difference=-0.00002,
            recent={
                "dev_15": -0.0001,
                "dev_16": 0.0001,
                "dev_17": -0.0001,
                "dev_18": -0.0001,
                "dev_19": 0.0001,
            },
        )
    )

    assert result["status"] == "RETAIN_GALL_DIVERSITY"
    assert result["recent5_wins"] == 3


def test_a60_rejects_target_regression_even_when_pooled_looks_better() -> None:
    result = _status(
        _comparison(
            pooled_difference=-0.00006,
            generator_all_difference=0.00001,
            recent={
                "dev_15": -0.0001,
                "dev_16": -0.0001,
                "dev_17": -0.0001,
                "dev_18": 0.0001,
                "dev_19": 0.0001,
            },
        )
    )

    assert result["status"] == "DO_NOT_RETAIN"

from gas_forecast.selection import choose_version


def _report(version: str, values: list[float]) -> dict[str, object]:
    names = ["dev_01", "dev_02", "blind"]
    return {
        "version": version,
        "mean_mape": sum(values) / len(values),
        "folds": [
            {
                "name": name,
                "mape": value,
                "validation_start": (
                    "2025-04-17 00:00:00" if name == "dev_02" else "2025-04-01 00:00:00"
                ),
                "validation_end": (
                    "2025-04-19 00:00:00" if name == "dev_02" else "2025-04-03 00:00:00"
                ),
                "by_target_horizon": {
                    "generator_1_t+15": value,
                    "generator_all_t+15": value,
                },
            }
            for name, value in zip(names, values)
        ],
    }


def test_version_selection_requires_majority_and_blind_not_worse() -> None:
    decision = choose_version(
        {
            "v1": _report("v1", [0.060, 0.061, 0.062]),
            "v2": _report("v2", [0.058, 0.059, 0.060]),
            "v25": _report("v25", [0.057, 0.058, 0.059]),
            "v3": _report("v3", [0.056, 0.057, 0.060]),
        }
    )
    assert decision["selected_version"] == "v25"
    assert decision["comparisons"]["v3_vs_v25"]["blind_not_worse"] is False
    assert decision["comparisons"]["v3_vs_v25"]["eligible_for_selection"] is True
    assert "v3_vs_v25" in decision["reason"]
    assert decision["report_summary"]["v1"]["folds"] == 3


def test_higher_version_is_still_diagnosed_when_predecessor_fails() -> None:
    decision = choose_version(
        {
            "v1": _report("v1", [0.060, 0.061, 0.062]),
            "v2": _report("v2", [0.058, 0.059, 0.060]),
            "v25": _report("v25", [0.059, 0.060, 0.061]),
            "v3": _report("v3", [0.057, 0.058, 0.059]),
        }
    )

    assert decision["selected_version"] == "v2"
    assert "v3_vs_v25" in decision["comparisons"]
    assert decision["comparisons"]["v3_vs_v25"]["accepted"] is True
    assert decision["comparisons"]["v3_vs_v25"]["eligible_for_selection"] is False

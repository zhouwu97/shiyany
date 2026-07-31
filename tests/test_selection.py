from gas_forecast.selection import choose_version


def _report(version: str, values: list[float]) -> dict[str, object]:
    names = ["dev_01", "dev_02", "blind"]
    return {
        "version": version,
        "mean_mape": sum(values) / len(values),
        "folds": [{"name": name, "mape": value} for name, value in zip(names, values)],
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

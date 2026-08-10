"""PRED-1 fail-closed schema 校验：重训前必须通过。

目标：任何 X3 Fold Replay 启动前，先生成当前代码+数据的 long_horizon
特征 schema，并与冻结的 X3 248 列 schema 逐列、逐序比对。不相等即
FAIL CLOSED，禁止重训。

同时复核 data_hash 与 feature_schema_hash 与 X3 实验收据一致。
"""

from __future__ import annotations

import argparse
import json
import hashlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from gas_forecast.config import forecast_config_from_dict
from gas_forecast.data import align_tables
from gas_forecast.experiments import dataframe_fingerprint, feature_schema_fingerprint
from gas_forecast.features import build_causal_features, load_price_schedule
from gas_forecast.rich_residual import RICH_FEATURE_GROUPS, rich_feature_config
from gas_forecast.rich_residual import select_rich_feature_columns

X3_DATA_HASH = "d9e5115d33a6c2df6e0d2c3adc9d4891c55131d4ee99a4c3f5d464194c5e6605"
X3_FEATURE_SCHEMA_HASH = "9aa17ad5496ab8c1fa9fd8142a84b666d7353eb61f6dde3c9c4ed4aeacc73136"
FROZEN_SCHEMA_PATH = Path(
    "results/raw/runs/audits/pred1_asset_audit_20260810/x3_feature_schema.json"
)


def _price_schedule(data_dir: Path):
    paths = sorted(data_dir.glob("*price*.xlsx"))
    if len(paths) > 1:
        raise ValueError(f"发现多个 price 文件: {paths}")
    return load_price_schedule(paths[0]) if paths else None


def _frozen_columns() -> list[str]:
    if not FROZEN_SCHEMA_PATH.exists():
        raise FileNotFoundError(f"缺少冻结 X3 schema: {FROZEN_SCHEMA_PATH}")
    payload = json.loads(FROZEN_SCHEMA_PATH.read_text(encoding="utf-8"))
    return list(payload["feature_columns"])


def check(
    data_dir: Path,
    config_path: Path,
) -> dict[str, Any]:
    config = forecast_config_from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    dataset = align_tables(data_dir, config.feature.frequency)
    effective = rich_feature_config(config, RICH_FEATURE_GROUPS, feature_profile="long_horizon")
    price = _price_schedule(data_dir)
    features = build_causal_features(dataset.frame, effective.feature, price)

    generated_columns = select_rich_feature_columns(features, "long_horizon")
    frozen_columns = _frozen_columns()

    data_hash = dataframe_fingerprint(dataset.frame)
    feature_schema_hash = feature_schema_fingerprint(features)
    generated_schema_sha = hashlib.sha256(
        json.dumps(generated_columns, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    frozen_schema_sha = hashlib.sha256(
        json.dumps(frozen_columns, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    result: dict[str, Any] = {
        "data_hash": data_hash,
        "data_hash_match": data_hash == X3_DATA_HASH,
        "feature_schema_hash": feature_schema_hash,
        "feature_schema_hash_match": feature_schema_hash == X3_FEATURE_SCHEMA_HASH,
        "generated_column_count": len(generated_columns),
        "frozen_column_count": len(frozen_columns),
        "generated_schema_sha256": generated_schema_sha,
        "frozen_schema_sha256": frozen_schema_sha,
        "schema_identical": generated_columns == frozen_columns,
        "schema_missing_columns": sorted(set(frozen_columns) - set(generated_columns)),
        "schema_extra_columns": sorted(set(generated_columns) - set(frozen_columns)),
        "fail_closed": (
            data_hash == X3_DATA_HASH
            and feature_schema_hash == X3_FEATURE_SCHEMA_HASH
            and generated_columns == frozen_columns
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/official/初赛-参赛者使用"))
    parser.add_argument("--config", type=Path, default=Path("results/raw/runs/audits/pred1_asset_audit_20260810/x3_config.json"))
    args = parser.parse_args()
    result = check(args.data_dir, args.config)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if not result["fail_closed"]:
        print("FAIL CLOSED: 生成 schema 与冻结 X3 schema 不一致，禁止重训。")
        raise SystemExit(1)
    print("PASS: 生成 schema == 冻结 X3 248 列 schema，可以进入 Gate B。")
    raise SystemExit(0)


if __name__ == "__main__":
    main()

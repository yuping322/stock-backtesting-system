import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.factor.merger import merge_factor_directory, merge_factor_files


def _write_csv(path, records):
    pd.DataFrame(records).to_csv(path, index=False)


def test_merge_factor_files_standardizes_column_aliases(tmp_path):
    file_a = tmp_path / "factors_a.csv"
    file_b = tmp_path / "factors_b.csv"

    _write_csv(
        file_a,
        [
            {"date": "2025-01-01", "stock_code": "000001", "factor_a": 1.0},
            {"date": "2025-01-02", "stock_code": "000001", "factor_a": 2.0},
        ],
    )
    _write_csv(
        file_b,
        [
            {"trade_date": "2025-01-01", "code": "1", "factor_b": 10.0},
            {"trade_date": "2025-01-02", "code": "000001", "factor_b": 20.0},
        ],
    )

    merged = merge_factor_files([str(file_a), str(file_b)], how="inner")

    assert merged.columns.tolist() == ["date", "stock_code", "factor_a", "factor_b"]
    assert merged["stock_code"].tolist() == ["000001", "000001"]
    assert merged["factor_b"].tolist() == [10.0, 20.0]


def test_merge_factor_directory_respects_pattern_and_excludes(tmp_path):
    task_dir = tmp_path / "task_20250101_000000"
    task_dir.mkdir()
    file_1 = task_dir / "factors_foo.csv"
    file_2 = task_dir / "factors_bar.csv"
    file_ignored = task_dir / "other.csv"

    _write_csv(
        file_1,
        [
            {"date": "2025-01-01", "stock_code": "000001", "factor_x": 1, "noise": 100},
        ],
    )
    _write_csv(
        file_2,
        [
            {"date": "2025-01-01", "stock_code": "000001", "factor_y": 2},
        ],
    )
    _write_csv(
        file_ignored,
        [
            {"date": "2025-01-01", "stock_code": "000001", "factor_z": 3},
        ],
    )

    output_file = tmp_path / "merged.csv"
    merged = merge_factor_directory(
        factor_dir=str(task_dir),
        pattern="factors_*.csv",
        output_file=str(output_file),
        exclude_factors=["noise"],
        how="outer",
    )

    assert output_file.exists()
    assert set(merged.columns) == {"date", "stock_code", "factor_x", "factor_y"}
    assert merged.loc[0, "factor_x"] == 1
    assert merged.loc[0, "factor_y"] == 2


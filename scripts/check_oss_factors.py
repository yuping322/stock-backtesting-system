import os
import sys
from pathlib import Path
from typing import List

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.data import factor_for_al, get_index_stocks

CONFIG_FILE = Path(__file__).resolve().parent.parent / "config" / "available_factors.txt"

START_DATE = "2025-11-25"
END_DATE = "2025-11-30"


def load_available_factors(limit: int = 10) -> List[str]:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"配置文件不存在: {CONFIG_FILE}")
    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        factors = [line.strip() for line in f if line.strip()]
    return factors[:limit]


def main():
    print("OSS 因子验证脚本")
    print(f"时间范围: {START_DATE} ~ {END_DATE}")

    stock_codes = get_index_stocks("small")[:3]
    if not stock_codes:
        raise RuntimeError("未能从 small 指数获取股票代码")

    print(f"使用股票: {stock_codes}")
    factors = load_available_factors(limit=15)

    for factor in factors:
        try:
            series = factor_for_al(stock_codes, START_DATE, END_DATE, factor)
            print(f"{factor}: {len(series)} 条数据")
        except Exception as exc:
            print(f"{factor}: 异常 -> {exc}")


def cli():
    try:
        main()
    except Exception as exc:
        print(f"验证失败: {exc}")


if __name__ == "__main__":
    cli()

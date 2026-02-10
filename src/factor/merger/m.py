from . import merge_factor_directory


merged = merge_factor_directory(
    factor_dir=str("/Users/fengzhi/Downloads/git/stock-backtesting-system/data/factor_tasks"),
    pattern="**/factors_*.csv",
    output_file=str("/Users/fengzhi/Downloads/git/stock-backtesting-system/data/merge_tasks/output_file"),
    exclude_factors=["noise"],
    how="outer",
)
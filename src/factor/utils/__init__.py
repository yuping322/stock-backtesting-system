"""
因子系统工具层

包含：
- 时间戳和目录管理
- 数据加载辅助
- 参数验证
- 常量定义
"""

from .helpers import (
    generate_timestamp,
    create_task_directory,
    get_stock_data_from_cache,
    normalize_stock_code,
    normalize_stock_codes,
    save_dataframe_to_csv,
    load_csv_to_dataframe,
    get_factor_output_path,
    get_metadata_output_path,
    get_readme_output_path,
)
from .validation import (
    validate_stock_codes,
    validate_date_range,
    validate_factor_names,
    validate_output_dir,
    validate_factor_file_path,
    validate_all_params,
)
from .constants import (
    BUILTIN_FACTORS,
    TALIB_FACTORS,
    OSS_FACTORS,
    DEFAULT_OUTPUT_DIR,
)

__all__ = [
    # helpers
    'generate_timestamp',
    'create_task_directory',
    'get_stock_data_from_cache',
    'normalize_stock_code',
    'normalize_stock_codes',
    'save_dataframe_to_csv',
    'load_csv_to_dataframe',
    'get_factor_output_path',
    'get_metadata_output_path',
    'get_readme_output_path',
    # validation
    'validate_stock_codes',
    'validate_date_range',
    'validate_factor_names',
    'validate_output_dir',
    'validate_factor_file_path',
    'validate_all_params',
    # constants
    'BUILTIN_FACTORS',
    'TALIB_FACTORS',
    'OSS_FACTORS',
    'DEFAULT_OUTPUT_DIR',
]

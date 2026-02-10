"""
常量定义模块

包含所有因子列表和配置常量
"""

# ========== 内置因子 ==========
BUILTIN_FACTORS = {
    'VOL10': {
        'name': 'VOL10',
        'description': '10日成交量比值',
        'type': 'builtin'
    },
    'RSI_14': {
        'name': 'RSI_14',
        'description': '14日相对强弱指标',
        'type': 'builtin'
    },
    'MA_20': {
        'name': 'MA_20',
        'description': '20日移动平均比值',
        'type': 'builtin'
    },
    'MACD_12_26_9': {
        'name': 'MACD_12_26_9',
        'description': 'MACD指标 (12, 26, 9)',
        'type': 'builtin'
    },
}

# ========== TA-Lib 因子 (常见的) ==========
TALIB_FACTORS = {
    # 趋势指标
    'SMA': 'Simple Moving Average',
    'EMA': 'Exponential Moving Average',
    'WMA': 'Weighted Moving Average',
    'DEMA': 'Double Exponential Moving Average',
    'TEMA': 'Triple Exponential Moving Average',
    'TRIMA': 'Triangular Moving Average',
    'KAMA': 'Kaufman Adaptive Moving Average',
    
    # 动量指标
    'RSI': 'Relative Strength Index',
    'STOCHRSI': 'Stochastic RSI',
    'MOM': 'Momentum',
    'ROC': 'Rate of Change',
    'ROCP': 'Rate of Change Percentage',
    'ROCR': 'Rate of Change Ratio',
    'TRIX': 'Triple Exponential Moving Average Oscillator',
    'WILLR': 'Williams\' %R',
    'CCI': 'Commodity Channel Index',
    'CMO': 'Chande Momentum Oscillator',
    'PPO': 'Percentage Price Oscillator',
    'APO': 'Absolute Price Oscillator',
    
    # 波动率指标
    'ATR': 'Average True Range',
    'NATR': 'Normalized Average True Range',
    'TRANGE': 'True Range',
    'ADX': 'Average Directional Index',
    'ADXR': 'Average Directional Index Rating',
    'DX': 'Directional Index',
    'PLUS_DI': 'Plus Directional Index',
    'PLUS_DM': 'Plus Directional Movement',
    'MINUS_DI': 'Minus Directional Index',
    'MINUS_DM': 'Minus Directional Movement',
    
    # 成交量指标
    'AD': 'Accumulation/Distribution Line',
    'ADOSC': 'Accumulation/Distribution Oscillator',
    'OBV': 'On Balance Volume',
    'MFI': 'Money Flow Index',
    
    # MACD
    'MACD': 'MACD',
    'STOCH': 'Stochastic',
    'BBANDS': 'Bollinger Bands',
    
    # 其他
    'AROON': 'Aroon',
    'AROONOSC': 'Aroon Oscillator',
}

# ========== OSS 因子 (Alpha158/360) ==========
# 前 20 个 Alpha158 因子（完整列表有 158 个）
ALPHA158_FACTORS = [
    'ALPHA158_001', 'ALPHA158_002', 'ALPHA158_003', 'ALPHA158_004', 'ALPHA158_005',
    'ALPHA158_006', 'ALPHA158_007', 'ALPHA158_008', 'ALPHA158_009', 'ALPHA158_010',
    'ALPHA158_011', 'ALPHA158_012', 'ALPHA158_013', 'ALPHA158_014', 'ALPHA158_015',
    'ALPHA158_016', 'ALPHA158_017', 'ALPHA158_018', 'ALPHA158_019', 'ALPHA158_020',
]

ALPHA360_FACTORS = [
    'ALPHA360_001', 'ALPHA360_002', 'ALPHA360_003', 'ALPHA360_004', 'ALPHA360_005',
    # ... 实际有 360 个因子
]

OSS_FACTORS = {
    'ALPHA158': {
        'description': 'QLib Alpha158 因子集',
        'count': 158,
        'factors': ALPHA158_FACTORS  # 示例，实际需要完整列表
    },
    'ALPHA360': {
        'description': 'QLib Alpha360 因子集',
        'count': 360,
        'factors': ALPHA360_FACTORS  # 示例，实际需要完整列表
    }
}

# ========== 文件输出配置 ==========
DEFAULT_OUTPUT_DIR = './data/factor_tasks'
DEFAULT_TASK_DIR_PATTERN = 'task_{timestamp}'
DEFAULT_FACTOR_FILE_PATTERN = 'factors_{timestamp}.csv'
DEFAULT_METADATA_FILE_PATTERN = 'task_metadata_{timestamp}.json'
DEFAULT_README_FILE_PATTERN = 'README_task_{timestamp}.md'

# ========== 数据加载配置 ==========
# 缓存位置
CACHE_DIR = './cache'

# 数据类型
DATA_TYPES = {
    'date': 'datetime64[ns]',
    'code': 'str',
    'stock_code': 'str',
    'open': 'float64',
    'high': 'float64',
    'low': 'float64',
    'close': 'float64',
    'volume': 'float64',
}

# ========== 日期格式 ==========
DATE_FORMAT = '%Y-%m-%d'
TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'

# ========== 因子生成配置 ==========
# 默认批处理大小（用于避免内存溢出）
DEFAULT_BATCH_SIZE = 100

# 数据加载超时时间（秒）
DATA_LOAD_TIMEOUT = 300

# ========== 日志配置 ==========
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

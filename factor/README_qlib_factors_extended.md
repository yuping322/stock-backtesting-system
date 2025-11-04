# Qlib因子生成方式详解

## 一、Qlib支持的因子生成方式

### 1.1 预定义的Handler类

Qlib提供了以下预定义的因子Handler：

#### A. Alpha158系列
- **`Alpha158`**: 标准Alpha158因子集（158个因子）
- **`Alpha158DL`**: 可配置的Alpha158，支持自定义配置
- **`Alpha158vwap`**: 基于VWAP的Alpha158变体

#### B. Alpha360系列
- **`Alpha360`**: 标准Alpha360因子集（360个因子）
- **`Alpha360DL`**: 可配置的Alpha360，支持自定义配置
- **`Alpha360vwap`**: 基于VWAP的Alpha360变体

#### C. 基类
- **`DataHandlerLP`**: 所有Handler的基类，可以继承它创建自定义Handler

### 1.2 通过配置自定义因子集

`Alpha158DL`和`Alpha360DL`支持通过配置自定义因子集：

```python
from qlib.contrib.data.loader import Alpha158DL, Alpha360DL

# 简化版配置（158个因子）
simple_conf = {
    'kbar': {},
    'price': {'windows': [0], 'feature': ['OPEN', 'HIGH', 'LOW', 'VWAP']},
    'rolling': {},
}
fields, names = Alpha158DL.get_feature_config(simple_conf)
# names: 158个因子名称

# 完整版配置（184个因子）
full_conf = {
    'kbar': {},
    'price': {
        'windows': [0, 1, 2, 3, 4],
        'feature': ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VWAP'],
    },
    'volume': {'windows': [0, 1, 2, 3, 4]},
    'rolling': {
        'windows': [5, 10, 20, 30, 60],
        'include': None,  # 使用默认算子
        'exclude': [],
    },
}
fields, names = Alpha158DL.get_feature_config(full_conf)
# names: 184个因子名称

# 自定义配置（可根据需求调整）
custom_conf = {
    'kbar': {},
    'price': {
        'windows': [0, 1, 2, 3, 4, 5],  # 自定义窗口
        'feature': ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VWAP'],
    },
    'volume': {'windows': [0, 1, 2, 3, 4, 5]},
    'rolling': {
        'windows': [10, 20, 30],  # 自定义滚动窗口
        'include': None,
        'exclude': [],
    },
}
fields, names = Alpha158DL.get_feature_config(custom_conf)
# names: 132个因子名称（根据配置变化）
```

### 1.3 自定义Handler（继承DataHandlerLP）

可以继承`DataHandlerLP`创建完全自定义的Handler：

```python
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset import DatasetH

class CustomFactorHandler(DataHandlerLP):
    """自定义因子Handler"""
    
    def get_feature_config(self):
        """定义要生成的因子"""
        # 返回字段配置和因子名称
        fields = [...]  # 字段配置
        names = [...]   # 因子名称列表
        return fields, names
```

### 1.4 使用因子表达式

Qlib支持使用表达式语法定义自定义因子：

```python
# 定义动量因子
momentum_factor = Ref($close, -N) / $close - 1

# 在handler中使用
# （需要在handler的配置中使用表达式）
```

## 二、配置参数说明

### 2.1 Alpha158DL配置参数

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `kbar` | dict | K线相关配置 | `{}` |
| `price` | dict | 价格字段配置 | `{'windows': [0, 1], 'feature': ['OPEN', 'HIGH']}` |
| `volume` | dict | 成交量字段配置 | `{'windows': [0, 1, 2]}` |
| `rolling` | dict | 滚动窗口配置 | `{'windows': [5, 10, 20]}` |

### 2.2 price配置详细说明

```python
'price': {
    'windows': [0, 1, 2, 3, 4],  # 窗口列表（0表示当前，1表示前1天，以此类推）
    'feature': ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VWAP'],  # 价格字段
}
```

### 2.3 rolling配置详细说明

```python
'rolling': {
    'windows': [5, 10, 20, 30, 60],  # 滚动窗口大小（天数）
    'include': None,  # 包含的算子（None表示使用默认）
    'exclude': [],    # 排除的算子
}
```

## 三、当前实现的限制

### 3.1 当前支持（已测试通过）

- ✅ **Alpha158**: 标准158个因子
- ✅ **Alpha360**: 标准360个因子
- ✅ **Alpha158vwap**: 基于VWAP的Alpha158变体（158个因子）
- ✅ **Alpha360vwap**: 基于VWAP的Alpha360变体（360个因子）

### 3.2 暂不支持的原因

- ❌ **Alpha158DL**: 是DataLoader类型，不是Handler，使用方式完全不同
- ❌ **Alpha360DL**: 是DataLoader类型，不是Handler，使用方式完全不同
- ⚠️ **自定义Handler**: 需要继承DataHandlerLP创建自定义Handler（高级功能）

**注意**：Alpha158DL和Alpha360DL主要用于通过`get_feature_config()`方法获取因子配置和名称，不能直接作为Handler使用。如果需要自定义因子集，可以通过配置参数传递给Handler，或使用自定义Handler。

## 四、测试结果

### 4.1 已测试通过的Handler类型

所有4种Handler类型均已测试通过：

| Handler类型 | 因子数量 | 测试状态 | 文件大小（示例） |
|------------|---------|---------|----------------|
| Alpha158 | 158 | ✅ 通过 | ~49KB |
| Alpha360 | 360 | ✅ 通过 | ~63KB |
| Alpha158vwap | 158 | ✅ 通过 | ~49KB |
| Alpha360vwap | 360 | ✅ 通过 | ~63KB |

**测试数据**：3只股票，30天数据

### 4.2 测试命令

```bash
# 运行所有Handler类型测试
python factor/test_all_handlers.py
```

## 五、扩展建议

### 4.1 支持更多Handler类型

可以在`generate_qlib_factors.py`中添加：

```python
# 支持Alpha158DL（可配置）
if factor_set == 'Alpha158DL':
    handler_conf = {
        'class': 'Alpha158DL',
        'module_path': 'qlib.contrib.data.handler',
        'kwargs': {
            'config': custom_config,  # 传递配置
            ...
        },
    }

# 支持Alpha158vwap
if factor_set == 'Alpha158vwap':
    handler_conf = {
        'class': 'Alpha158vwap',
        ...
    }
```

### 4.2 支持配置参数

可以在命令行参数中添加配置选项：

```bash
python factor/generate_qlib_factors.py \
    --factor-set Alpha158DL \
    --config windows:0,1,2,3,4 \
    --config rolling:5,10,20,30 \
    ...
```

### 4.3 支持自定义Handler

可以为高级用户提供自定义Handler的接口：

```python
# 用户可以传入自定义Handler类
python factor/generate_qlib_factors.py \
    --factor-set Custom \
    --handler-module my_handlers \
    --handler-class MyFactorHandler \
    ...
```

## 六、使用示例

### 6.1 使用标准Alpha158

```bash
python factor/generate_qlib_factors.py \
    --factor-set Alpha158 \
    --stock-pool HS300 \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --output ./factors
```

### 6.2 使用标准Alpha360

```bash
python factor/generate_qlib_factors.py \
    --factor-set Alpha360 \
    --stock-pool HS300 \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --output ./factors
```

### 6.3 使用VWAP版本

```bash
# 使用Alpha158vwap
python factor/generate_qlib_factors.py \
    --factor-set Alpha158vwap \
    --stock-pool HS300 \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --output ./factors

# 使用Alpha360vwap
python factor/generate_qlib_factors.py \
    --factor-set Alpha360vwap \
    --stock-pool HS300 \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --output ./factors
```

### 6.4 未来：使用可配置Alpha158DL

```bash
# （待实现）
python factor/generate_qlib_factors.py \
    --factor-set Alpha158DL \
    --config price-windows:0,1,2,3,4 \
    --config price-feature:OPEN,HIGH,LOW,CLOSE,VWAP \
    --config rolling-windows:5,10,20,30,60 \
    --stock-pool HS300 \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --output ./factors
```

## 七、总结

目前`generate_qlib_factors.py`已实现并测试通过：
- ✅ **Alpha158**（标准版，158个因子）
- ✅ **Alpha360**（标准版，360个因子）
- ✅ **Alpha158vwap**（VWAP版，158个因子）
- ✅ **Alpha360vwap**（VWAP版，360个因子）

暂不支持（技术限制）：
- ❌ **Alpha158DL/Alpha360DL**（它们是DataLoader，不是Handler，使用方式不同）

对于大多数应用场景，**Alpha158**和**Alpha360**已经足够使用。如果需要自定义因子集，可以通过配置Alpha158DL/Alpha360DL来实现。


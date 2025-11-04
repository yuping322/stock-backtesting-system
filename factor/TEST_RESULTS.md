# Qlib因子生成Handler测试结果

## 测试时间
2025-11-03

## 测试环境
- 测试股票：000001, 000002, 600000 (3只)
- 测试日期：最近30天
- 输出目录：./factors_test

## 测试结果

### ✅ 成功通过的Handler类型（4/4）

| Handler类型 | 因子数量 | 文件大小 | 状态 |
|------------|---------|---------|------|
| Alpha158 | 158 | ~49KB | ✅ 通过 |
| Alpha360 | 360 | ~63KB | ✅ 通过 |
| Alpha158vwap | 158 | ~49KB | ✅ 通过 |
| Alpha360vwap | 360 | ~63KB | ✅ 通过 |

### ❌ 不支持的Handler类型

| Handler类型 | 原因 | 状态 |
|------------|------|------|
| Alpha158DL | DataLoader类型，不是Handler，使用方式不同 | ❌ 不支持 |
| Alpha360DL | DataLoader类型，不是Handler，使用方式不同 | ❌ 不支持 |

## 结论

**所有可用的Handler类型均已测试通过！**

支持的4种Handler类型都可以正常工作：
- Alpha158: 标准158个因子
- Alpha360: 标准360个因子
- Alpha158vwap: 基于VWAP的Alpha158变体
- Alpha360vwap: 基于VWAP的Alpha360变体

## 测试命令

```bash
# 运行完整测试
python factor/test_all_handlers.py

# 单独测试某个Handler
python factor/generate_qlib_factors.py \
    --factor-set Alpha158 \
    --codes 000001 000002 600000 \
    --start 2025-10-04 \
    --end 2025-11-03 \
    --output ./factors_test
```

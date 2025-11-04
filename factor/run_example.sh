#!/bin/bash
# Qlib因子生成与验证示例脚本

cd "$(dirname "$0")/.."

echo "========================================"
echo "步骤1: 生成最近3个月的Alpha158因子文件"
echo "========================================"

# 计算日期范围（最近3个月）
END_DATE=$(date +%Y-%m-%d)
START_DATE=$(date -v-90d +%Y-%m-%d 2>/dev/null || date -d "90 days ago" +%Y-%m-%d)

echo "日期范围: $START_DATE 到 $END_DATE"

# 使用默认股票列表（确保可以运行）
STOCKS="000001 000002 600000 600519 000858"

echo "使用股票: $STOCKS"

# 生成因子文件
python factor/generate_qlib_factors.py \
    --factor-set Alpha158 \
    --codes $STOCKS \
    --start $START_DATE \
    --end $END_DATE \
    --output ./factors \
    --rebuild

if [ $? -ne 0 ]; then
    echo "生成因子文件失败，请检查错误信息"
    exit 1
fi

echo ""
echo "========================================"
echo "步骤2: 验证因子"
echo "========================================"

# 查找生成的因子文件
FACTOR_FILE=$(ls -t factors/Alpha158_*.csv 2>/dev/null | head -1)

if [ -z "$FACTOR_FILE" ]; then
    echo "未找到因子文件，请检查生成步骤"
    exit 1
fi

echo "使用因子文件: $FACTOR_FILE"

# 获取因子名称（从文件读取前几行）
FACTOR_NAMES=$(python -c "
import pandas as pd
df = pd.read_csv('$FACTOR_FILE', nrows=1)
factors = [col for col in df.columns if col not in ['date', 'code']]
# 选择前3个因子进行测试
test_factors = factors[:3] if len(factors) >= 3 else factors
print(' '.join(test_factors))
" 2>/dev/null)

if [ -z "$FACTOR_NAMES" ]; then
    echo "无法读取因子名称，使用默认因子"
    FACTOR_NAMES="ROC5 MA10 STD5"
fi

echo "验证因子: $FACTOR_NAMES"

# 运行验证（使用factor.py）
python factor/factor.py \
    --start $START_DATE \
    --end $END_DATE \
    --stock-pool 000300 \
    --factors $FACTOR_NAMES \
    --quantiles 5 \
    --periods 5 10 \
    --factor-dir ./factors

echo ""
echo "========================================"
echo "完成！"
echo "========================================"


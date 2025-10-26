#!/bin/bash
# 测试指定范围的因子

# 用法: ./run_factor_range.sh START_INDEX END_INDEX
# 例如: ./run_factor_range.sh 1 100  (测试第1到100个因子)

if [ $# -lt 2 ]; then
    echo "用法: $0 START_INDEX END_INDEX"
    echo "例如: $0 1 100  (测试第1到100个因子)"
    echo ""
    echo "可用因子总数: $(cat available_factors.txt | wc -l)"
    exit 1
fi

START_INDEX=$1
END_INDEX=$2

# 日期范围
START_DATE="2024-07-26"
END_DATE="2024-10-24"
OUTPUT_DIR="results/factor_test_${START_INDEX}_${END_INDEX}"

# 从 available_factors.txt 读取指定范围的因子
if [ ! -f "available_factors.txt" ]; then
    echo "❌ 错误: available_factors.txt 不存在"
    exit 1
fi

FACTORS=$(sed -n "${START_INDEX},${END_INDEX}p" available_factors.txt | tr '\n' ' ')
FACTOR_COUNT=$(echo "$FACTORS" | wc -w)

echo "======================================================================"
echo "测试因子范围: $START_INDEX - $END_INDEX"
echo "======================================================================"
echo "因子数量: $FACTOR_COUNT"
echo "日期范围: $START_DATE ~ $END_DATE"
echo "输出目录: $OUTPUT_DIR"
echo "======================================================================"
echo ""

# 运行测试
python main_factor.py \
  --factors $FACTORS \
  --start $START_DATE \
  --end $END_DATE \
  --output-dir $OUTPUT_DIR \
  --plot true \
  --plot-mode save

echo ""
echo "✅ 测试完成！"
echo "📊 查看报告: cat $OUTPUT_DIR/README.md"


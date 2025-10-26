#!/bin/bash
# 测试前50个因子（快速验证）

# 日期范围
START_DATE="2024-07-26"
END_DATE="2024-10-24"
OUTPUT_DIR="results/factor_test_first50"

# 从 available_factors.txt 读取前50个因子
if [ ! -f "available_factors.txt" ]; then
    echo "❌ 错误: available_factors.txt 不存在"
    exit 1
fi

FACTORS=$(head -50 available_factors.txt | tr '\n' ' ')

echo "======================================================================"
echo "测试前50个因子"
echo "======================================================================"
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


#!/bin/bash
# 测试所有 OSS 因子（最近3个月）

# 日期范围
START_DATE="2025-07-26"
END_DATE="2025-10-24"
OUTPUT_DIR="results/factor_test_all_oss"

# 从 available_factors.txt 读取所有因子
if [ ! -f "available_factors.txt" ]; then
    echo "❌ 错误: available_factors.txt 不存在"
    echo "请先运行: python get_all_factors.py"
    exit 1
fi

# 读取因子列表（每行一个因子）
FACTORS=$(cat available_factors.txt | tr '\n' ' ')
FACTOR_COUNT=$(cat available_factors.txt | wc -l)

echo "======================================================================"
echo "准备测试所有 OSS 因子"
echo "======================================================================"
echo "因子总数: $FACTOR_COUNT"
echo "日期范围: $START_DATE ~ $END_DATE"
echo "输出目录: $OUTPUT_DIR"
echo "======================================================================"
echo ""

# 确认运行
read -p "确认要测试 $FACTOR_COUNT 个因子吗？(y/N): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "取消测试"
    exit 0
fi

echo ""
echo "开始测试..."
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
echo "======================================================================"
echo "✅ 测试完成！"
echo "======================================================================"
echo "结果保存在: $OUTPUT_DIR"
echo "查看报告: cat $OUTPUT_DIR/README.md"
echo "查看汇总: cat $OUTPUT_DIR/summary.csv"
echo "======================================================================"

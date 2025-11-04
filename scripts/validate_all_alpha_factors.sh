#!/bin/bash
# 验证所有Alpha158和Alpha360因子

cd "$(dirname "$0")/.."

echo "============================================================"
echo "验证所有Alpha因子（Alpha158 + Alpha360）"
echo "============================================================"

# 日期范围（参考run_all_factors.sh）
START_DATE="2025-07-26"
END_DATE="2025-10-24"
OUTPUT_DIR="results/validate_all_alpha_factors"

# 读取因子文件
ALPHA158_FILE="alpha158_factors.txt"
ALPHA360_FILE="alpha360_factors.txt"

if [ ! -f "$ALPHA158_FILE" ]; then
    echo "❌ 错误: $ALPHA158_FILE 不存在"
    echo "   请先运行: python scripts/generate_factor_lists.py"
    exit 1
fi

if [ ! -f "$ALPHA360_FILE" ]; then
    echo "❌ 错误: $ALPHA360_FILE 不存在"
    echo "   请先运行: python scripts/generate_factor_lists.py"
    exit 1
fi

# 提取因子名称（跳过注释行和空行）
ALPHA158_FACTORS=$(grep -v "^#" "$ALPHA158_FILE" | grep -v "^$" | grep -v "===" | tr '\n' ' ')
ALPHA360_FACTORS=$(grep -v "^#" "$ALPHA360_FILE" | grep -v "^$" | grep -v "===" | tr '\n' ' ')

# 统计因子数量
ALPHA158_COUNT=$(echo "$ALPHA158_FACTORS" | wc -w | tr -d ' ')
ALPHA360_COUNT=$(echo "$ALPHA360_FACTORS" | wc -w | tr -d ' ')
TOTAL_COUNT=$((ALPHA158_COUNT + ALPHA360_COUNT))

echo ""
echo "📊 因子统计:"
echo "   Alpha158: $ALPHA158_COUNT 个因子"
echo "   Alpha360: $ALPHA360_COUNT 个因子"
echo "   总计: $TOTAL_COUNT 个因子"
echo ""
echo "日期范围: $START_DATE ~ $END_DATE"
echo "输出目录: $OUTPUT_DIR"
echo "============================================================"
echo ""

# 确认运行
read -p "确认要验证 $TOTAL_COUNT 个因子吗？(y/N): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "取消验证"
    exit 0
fi

echo ""
echo "开始验证..."
echo ""

# 合并所有因子
ALL_FACTORS="$ALPHA158_FACTORS $ALPHA360_FACTORS"

# 因子文件目录（如果存在）
FACTOR_DIR="factors"

# 调用main_factor.py进行验证（参考run_all_factors.sh的参数）
python main_factor.py \
  --factors $ALL_FACTORS \
  --start $START_DATE \
  --end $END_DATE \
  --stock-pool small \
  --quantiles 5 \
  --periods 5 10 \
  --roll-win 20 \
  --output-dir $OUTPUT_DIR \
  --plot false \
  --factor-dir $FACTOR_DIR \
  --monitor-csv "$OUTPUT_DIR/monitor.csv"

EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 因子验证完成！"
    echo "============================================================"
    echo "结果保存在: $OUTPUT_DIR"
    echo "查看汇总: cat $OUTPUT_DIR/summary.csv"
    echo "查看监控: cat $OUTPUT_DIR/monitor.csv"
else
    echo "❌ 因子验证失败 (退出码: $EXIT_CODE)"
fi
echo "============================================================"

exit $EXIT_CODE


#!/bin/bash
# 测试所有 TA-Lib 技术指标因子（最近3个月）

# 日期范围
START_DATE="2025-04-26"
END_DATE="2025-10-24"
OUTPUT_DIR="results/factor_test_all_talib"
MAX_STOCKS=1000  # 限制股票数量，避免内存和时间问题

# 从 talib_factors.txt 读取所有 TA-Lib 因子
if [ ! -f "talib_factors.txt" ]; then
    echo "❌ 错误: talib_factors.txt 不存在"
    echo "请先运行: python generate_talib_factors.py"
    exit 1
fi

# 读取因子列表（跳过注释行，每行一个因子）
ALL_FACTORS=$(grep -v '^#' talib_factors.txt | tr '\n' ' ')
ALL_FACTOR_COUNT=$(grep -v '^#' talib_factors.txt | wc -l)

# 检查已存在的因子结果
echo "检查已存在的因子结果..."
FACTORS_TO_RUN=""
SKIPPED_COUNT=0

for factor in $ALL_FACTORS; do
    if [ -d "$OUTPUT_DIR/$factor" ] && [ -f "$OUTPUT_DIR/$factor/README.md" ]; then
        echo "  ⏭️  跳过已存在的因子: $factor"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    else
        FACTORS_TO_RUN="$FACTORS_TO_RUN $factor"
    fi
done

# 去除开头的空格
FACTORS_TO_RUN=$(echo $FACTORS_TO_RUN | sed 's/^ *//')

# 计算要运行的因子数量
FACTOR_COUNT=$(echo $FACTORS_TO_RUN | wc -w)

echo "======================================================================"
echo "准备测试所有 TA-Lib 技术指标因子"
echo "======================================================================"
echo "因子总数: $ALL_FACTOR_COUNT"
echo "已存在: $SKIPPED_COUNT"
echo "待运行: $FACTOR_COUNT"
echo "日期范围: $START_DATE ~ $END_DATE"
echo "输出目录: $OUTPUT_DIR"
echo "======================================================================"
echo ""

# 如果没有需要运行的因子，直接退出
if [ $FACTOR_COUNT -eq 0 ]; then
    echo "✅ 所有因子都已存在，无需运行"
    echo ""
    echo "======================================================================"
    echo "结果保存在: $OUTPUT_DIR"
    echo "查看报告: cat $OUTPUT_DIR/README.md"
    echo "查看汇总: cat $OUTPUT_DIR/summary.csv"
    echo "======================================================================"
    exit 0
fi

# 确认运行
read -p "确认要测试 $FACTOR_COUNT 个 TA-Lib 因子吗？(y/N): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "取消测试"
    exit 0
fi

echo ""
echo "开始测试..."
echo ""

# 运行测试
python main_factor.py \
  --factors $FACTORS_TO_RUN \
  --start $START_DATE \
  --end $END_DATE \
  --output-dir $OUTPUT_DIR \
  --max-stocks $MAX_STOCKS \
  --plot true \
  --plot-mode save

echo ""
echo "======================================================================"
echo "✅ TA-Lib 因子测试完成！"
echo "======================================================================"
echo "结果保存在: $OUTPUT_DIR"
echo "查看报告: cat $OUTPUT_DIR/README.md"
echo "查看汇总: cat $OUTPUT_DIR/summary.csv"
echo "======================================================================"
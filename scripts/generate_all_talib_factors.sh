#!/bin/bash
# 生成所有 TA-Lib 技术指标因子数据（2024全年）

# 日期范围
START_DATE="2025-01-01"
END_DATE="2025-11-23"
OUTPUT_DIR="data/talib_factors_2025"
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

# 检查已存在的因子数据文件
echo "检查已存在的因子数据文件..."
FACTORS_TO_RUN=""
SKIPPED_COUNT=0

for factor in $ALL_FACTORS; do
    if [ -f "$OUTPUT_DIR/${factor}_2025-01-01_2025-11-23.csv" ]; then
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
echo "准备生成所有 TA-Lib 技术指标因子数据"
echo "======================================================================"
echo "因子总数: $ALL_FACTOR_COUNT"
echo "已存在: $SKIPPED_COUNT"
echo "待生成: $FACTOR_COUNT"
echo "日期范围: $START_DATE ~ $END_DATE"
echo "输出目录: $OUTPUT_DIR"
echo "======================================================================"
echo ""

# 如果没有需要生成的因子，直接退出
if [ $FACTOR_COUNT -eq 0 ]; then
    echo "✅ 所有因子数据都已存在，无需生成"
    echo ""
    echo "======================================================================"
    echo "因子数据保存在: $OUTPUT_DIR"
    echo "查看汇总报告: cat $OUTPUT_DIR/generation_summary.txt"
    echo "======================================================================"
    exit 0
fi

# 确认运行
read -p "确认要生成 $FACTOR_COUNT 个 TA-Lib 因子数据吗？(y/N): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "取消生成"
    exit 0
fi

echo ""
echo "开始生成因子数据..."
echo ""

# 运行因子数据生成
python generate_factors.py \
  --factors $FACTORS_TO_RUN \
  --start $START_DATE \
  --end $END_DATE \
  --output-dir $OUTPUT_DIR \
  --max-stocks $MAX_STOCKS

echo ""
echo "======================================================================"
echo "✅ TA-Lib 因子数据生成完成！"
echo "======================================================================"
echo "因子数据保存在: $OUTPUT_DIR"
echo "查看汇总报告: cat $OUTPUT_DIR/generation_summary.txt"
echo "======================================================================"</content>
<parameter name="filePath">/Users/fengzhi/Downloads/git/stock-backtesting-system/scripts/generate_all_talib_factors.sh
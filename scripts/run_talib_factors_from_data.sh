#!/bin/bash
# 测试所有已生成的 TA-Lib 技术指标因子（基于data/talib_factors_2025）

# 配置参数
FACTOR_DIR="data/talib_factors_2025"
OUTPUT_DIR="results/factor_test_talib_202509"
START_DATE="2025-09-01"
END_DATE="2025-11-23"
MAX_STOCKS=1000  # 限制股票数量，避免内存和时间问题

# 检查因子目录是否存在
if [ ! -d "$FACTOR_DIR" ]; then
    echo "❌ 错误: 因子目录不存在: $FACTOR_DIR"
    echo "请先运行因子生成脚本生成因子数据"
    exit 1
fi

# 从因子目录中提取所有因子名称
echo "扫描因子目录: $FACTOR_DIR"
ALL_FACTOR_FILES=$(ls $FACTOR_DIR/*.csv 2>/dev/null | grep -v generation_summary.txt)

if [ -z "$ALL_FACTOR_FILES" ]; then
    echo "❌ 错误: 在 $FACTOR_DIR 中未找到任何因子CSV文件"
    exit 1
fi

# 提取因子名称（从文件名中解析）
ALL_FACTORS=""
for file in $ALL_FACTOR_FILES; do
    # 从文件名提取因子名，例如: TALIB_AD_2025-01-01_2025-11-23.csv -> TALIB_AD
    filename=$(basename "$file" .csv)
    # 去掉日期部分，保留因子名
    factor_name=$(echo "$filename" | sed 's/_[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}_[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}$//')
    ALL_FACTORS="$ALL_FACTORS $factor_name"
done

# 去除重复的因子名
ALL_FACTORS=$(echo $ALL_FACTORS | tr ' ' '\n' | sort | uniq | tr '\n' ' ')
ALL_FACTOR_COUNT=$(echo $ALL_FACTORS | wc -w)

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
echo "准备测试已生成的 TA-Lib 技术指标因子"
echo "======================================================================"
echo "因子目录: $FACTOR_DIR"
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
  --factor-dir $FACTOR_DIR \
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
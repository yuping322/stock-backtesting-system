#!/bin/bash
# 继续运行剩余因子（跳过已完成的）

# 配置
START_DATE="2025-07-26"
END_DATE="2025-10-24"
OUTPUT_DIR="results/factor_test_all_oss"

# 检查必要文件
if [ ! -f "available_factors.txt" ]; then
    echo "❌ 错误: available_factors.txt 不存在"
    exit 1
fi

# 获取所有因子
ALL_FACTORS=$(cat available_factors.txt | tr '\n' ' ')
TOTAL_COUNT=$(cat available_factors.txt | wc -l | tr -d ' ')

# 获取已完成的因子（有README.md的文件夹）
COMPLETED_FACTORS=""
COMPLETED_COUNT=0

if [ -d "$OUTPUT_DIR" ]; then
    for factor_dir in "$OUTPUT_DIR"/*/; do
        if [ -d "$factor_dir" ]; then
            factor_name=$(basename "$factor_dir")
            # 检查是否有 README.md 或任何 period_ 文件夹（说明已处理过）
            if [ -f "$factor_dir/README.md" ] || ls "$factor_dir"/period_* &>/dev/null; then
                COMPLETED_FACTORS="$COMPLETED_FACTORS $factor_name"
                COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
            fi
        fi
    done
fi

# 计算剩余因子
REMAINING_FACTORS=""
for factor in $ALL_FACTORS; do
    # 检查该因子是否在已完成列表中
    if [[ ! " $COMPLETED_FACTORS " =~ " $factor " ]]; then
        REMAINING_FACTORS="$REMAINING_FACTORS $factor"
    fi
done

REMAINING_COUNT=$(echo $REMAINING_FACTORS | wc -w | tr -d ' ')

echo "======================================================================"
echo "因子测试恢复脚本"
echo "======================================================================"
echo "因子总数:   $TOTAL_COUNT"
echo "已完成:     $COMPLETED_COUNT"
echo "待处理:     $REMAINING_COUNT"
echo "日期范围:   $START_DATE ~ $END_DATE"
echo "输出目录:   $OUTPUT_DIR"
echo "======================================================================"
echo ""

if [ $REMAINING_COUNT -eq 0 ]; then
    echo "✅ 所有因子已完成！"
    echo ""
    echo "查看结果："
    echo "  cat $OUTPUT_DIR/README.md"
    echo "  cat $OUTPUT_DIR/summary.csv"
    exit 0
fi

# 显示已完成的因子
if [ $COMPLETED_COUNT -gt 0 ]; then
    echo "已完成的因子 ($COMPLETED_COUNT 个):"
    echo "$COMPLETED_FACTORS" | tr ' ' '\n' | head -20
    if [ $COMPLETED_COUNT -gt 20 ]; then
        echo "... 还有 $((COMPLETED_COUNT - 20)) 个"
    fi
    echo ""
fi

# 显示待处理的因子
echo "待处理的因子 ($REMAINING_COUNT 个):"
echo "$REMAINING_FACTORS" | tr ' ' '\n' | head -10
if [ $REMAINING_COUNT -gt 10 ]; then
    echo "... 还有 $((REMAINING_COUNT - 10)) 个"
fi
echo ""

# 确认运行
read -p "确认要继续测试剩余 $REMAINING_COUNT 个因子吗？(y/N): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "取消测试"
    exit 0
fi

echo ""
echo "开始测试剩余因子..."
echo ""

# 运行测试
python main_factor.py \
  --factors $REMAINING_FACTORS \
  --start $START_DATE \
  --end $END_DATE \
  --output-dir $OUTPUT_DIR \
  --plot true \
  --plot-mode save

EXIT_CODE=$?

echo ""
echo "======================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 测试完成！"
else
    echo "⚠️  测试过程中出现错误（退出码: $EXIT_CODE）"
    echo "已完成的因子结果已保存，可以重新运行本脚本继续"
fi
echo "======================================================================"
echo "结果保存在: $OUTPUT_DIR"
echo "查看报告: cat $OUTPUT_DIR/README.md"
echo "查看汇总: cat $OUTPUT_DIR/summary.csv"
echo "======================================================================"


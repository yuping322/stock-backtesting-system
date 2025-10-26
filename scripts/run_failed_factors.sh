#!/bin/bash
# 重新运行失败的因子（删除空文件夹后重新测试）

# 配置
START_DATE="2024-07-26"
END_DATE="2024-10-24"
OUTPUT_DIR="results/factor_test_all_oss"

# 检查必要文件
if [ ! -f "available_factors.txt" ]; then
    echo "❌ 错误: available_factors.txt 不存在"
    exit 1
fi

echo "======================================================================"
echo "找出失败的因子（空文件夹）"
echo "======================================================================"

# 找出失败的因子（没有README.md和period_文件夹的）
FAILED_FACTORS=""
FAILED_COUNT=0

for dir in "$OUTPUT_DIR"/*/; do
    if [ -d "$dir" ]; then
        factor_name=$(basename "$dir")
        # 检查是否有 README.md 或任何 period_ 文件夹
        if [ ! -f "$dir/README.md" ] && [ ! -d "$dir/period_5" ]; then
            FAILED_FACTORS="$FAILED_FACTORS $factor_name"
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
    fi
done

if [ $FAILED_COUNT -eq 0 ]; then
    echo "✅ 没有失败的因子！"
    exit 0
fi

echo "找到 $FAILED_COUNT 个失败的因子"
echo ""
echo "失败的因子列表:"
echo "$FAILED_FACTORS" | tr ' ' '\n' | head -20
if [ $FAILED_COUNT -gt 20 ]; then
    echo "... 还有 $((FAILED_COUNT - 20)) 个"
fi
echo ""

# 确认删除并重新运行
read -p "确认要删除这些空文件夹并重新测试吗？(y/N): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "取消操作"
    exit 0
fi

echo ""
echo "正在删除空文件夹..."
for dir in "$OUTPUT_DIR"/*/; do
    if [ -d "$dir" ]; then
        factor_name=$(basename "$dir")
        # 检查是否有 README.md 或任何 period_ 文件夹
        if [ ! -f "$dir/README.md" ] && [ ! -d "$dir/period_5" ]; then
            echo "  删除: $dir"
            rm -rf "$dir"
        fi
    fi
done

echo ""
echo "开始重新测试失败的因子..."
echo ""

# 运行测试
python main_factor.py \
  --factors $FAILED_FACTORS \
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


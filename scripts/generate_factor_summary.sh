#!/usr/bin/env bash
# 生成因子汇总，遍历每个因子文件夹读取README

BATCH_DIR="results/factor_test_all_oss"
OUTPUT_FILE="results/factor_test_all_oss/ALL_FACTORS_SUMMARY.md"

echo "======================================================================"
echo "生成因子汇总报告"
echo "======================================================================"
echo "批次目录: $BATCH_DIR"
echo "输出文件: $OUTPUT_FILE"
echo "======================================================================"

# 创建汇总文档头部
cat > "$OUTPUT_FILE" << EOF
# 所有因子测试结果汇总

生成时间: $(date '+%Y-%m-%d %H:%M:%S')

本文件汇总了 \`$BATCH_DIR\` 目录下所有因子的测试结果。

EOF

# 统计变量
TOTAL_FACTORS=0
TOTAL_FILES=0

# 遍历所有因子文件夹
for factor_dir in "$BATCH_DIR"/*/; do
    if [ ! -d "$factor_dir" ]; then
        continue
    fi
    
    factor_name=$(basename "$factor_dir")
    
    # 跳过非因子目录
    if [ "$factor_name" = "all" ] || [ "$factor_name" = "all_plots_test" ]; then
        continue
    fi
    
    # 检查是否有README.md
    if [ -f "$factor_dir/README.md" ]; then
        echo "处理因子: $factor_name"
        
        echo "" >> "$OUTPUT_FILE"
        echo "---" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        echo "## 因子: $factor_name" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        
        # 读取因子README
        cat "$factor_dir/README.md" >> "$OUTPUT_FILE"
        
        echo "" >> "$OUTPUT_FILE"
        
        TOTAL_FACTORS=$((TOTAL_FACTORS + 1))
        TOTAL_FILES=$((TOTAL_FILES + 1))
    fi
done

# 添加汇总统计
cat >> "$OUTPUT_FILE" << EOF

---

## 汇总统计

- **处理因子总数**: $TOTAL_FACTORS
- **读取文件总数**: $TOTAL_FILES
- **目录**: \`$BATCH_DIR\`

## 查看详细结果

每个因子的详细结果位于各自的文件夹中：
\`\`\`
$BATCH_DIR/[因子名]/README.md
\`\`\`

EOF

echo ""
echo "✅ 汇总完成！"
echo "📄 文件位置: $OUTPUT_FILE"
echo "📊 因子数: $TOTAL_FACTORS"
echo ""
echo "查看汇总:"
echo "  cat $OUTPUT_FILE"
echo ""


#!/usr/bin/env bash
# 生成汇总 README，整合所有因子测试结果

OUTPUT_FILE="results/ALL_RESULTS_SUMMARY.md"

echo "======================================================================"
echo "生成因子测试结果汇总"
echo "======================================================================"

# 创建汇总文档
cat > "$OUTPUT_FILE" << EOF
# 因子测试结果汇总

生成时间: $(date '+%Y-%m-%d %H:%M:%S')

本文件汇总了所有因子测试批次的结果。

## 测试批次

EOF

# 遍历所有结果目录
TOTAL_FACTORS=0
TOTAL_DIRS=0

for dir in results/*/; do
    if [ ! -d "$dir" ]; then
        continue
    fi
    
    dirname=$(basename "$dir")
    
    # 跳过特定目录
    if [ "$dirname" = "all" ] || [ "$dirname" = "all_plots_test" ]; then
        continue
    fi
    
    # 检查是否有README.md
    if [ -f "$dir/README.md" ]; then
        echo "处理: $dirname"
        
        # 统计因子数量
        factor_count=$(find "$dir" -maxdepth 1 -type d | grep -v "^$dir$" | wc -l | tr -d ' ')
        
        if [ "$factor_count" -gt 0 ]; then
            echo "" >> "$OUTPUT_FILE"
            echo "### 批次: $dirname" >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"
            echo "- 因子数量: $factor_count" >> "$OUTPUT_FILE"
            echo "- 目录路径: \`$dir\`" >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"
            echo "详细内容:" >> "$OUTPUT_FILE"
            echo "\`\`\`" >> "$OUTPUT_FILE"
            
            # 提取README的完整内容（读取整个文件）
            cat "$dir/README.md" >> "$OUTPUT_FILE"
            
            echo "\`\`\`" >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"
            
            TOTAL_FACTORS=$((TOTAL_FACTORS + factor_count))
            TOTAL_DIRS=$((TOTAL_DIRS + 1))
        fi
    fi
done

# 添加汇总信息
cat >> "$OUTPUT_FILE" << EOF

## 汇总统计

- **测试批次总数**: $TOTAL_DIRS
- **因子测试总数**: $TOTAL_FACTORS

## 查看详细结果

EOF

# 添加各批次的链接
for dir in results/*/; do
    if [ ! -d "$dir" ]; then
        continue
    fi
    
    dirname=$(basename "$dir")
    
    if [ "$dirname" == "all" ] || [ "$dirname" == "all_plots_test" ]; then
        continue
    fi
    
    if [ -f "$dir/README.md" ]; then
        echo "- [$dirname]($dir/README.md)" >> "$OUTPUT_FILE"
    fi
done

# 日期已经在创建时替换了，无需再次替换

echo ""
echo "✅ 汇总完成！"
echo "📄 文件位置: $OUTPUT_FILE"
echo "📊 批次数: $TOTAL_DIRS"
echo "🔢 因子数: $TOTAL_FACTORS"
echo ""
echo "查看汇总:"
echo "  cat $OUTPUT_FILE"
echo ""


#!/usr/bin/env bash
# 生成更好的因子汇总格式，按周期分组便于比较

BATCH_DIR="results/factor_test_all_oss"
OUTPUT_FILE="results/factor_test_all_oss/ALL_FACTORS_SUMMARY_FORMATTED.md"

echo "======================================================================"
echo "生成格式化因子汇总报告"
echo "======================================================================"
echo "批次目录: $BATCH_DIR"
echo "输出文件: $OUTPUT_FILE"
echo "======================================================================"

# 临时文件
SUMMARY_FILE=$(mktemp)
ALIVE_FILE=$(mktemp)
WARNING_FILE=$(mktemp)
DEAD_FILE=$(mktemp)

# 统计变量
TOTAL_FACTORS=0
ALIVE_COUNT=0
WARNING_COUNT=0
DEAD_COUNT=0

# 遍历所有因子，分类收集
for factor_dir in "$BATCH_DIR"/*/; do
    if [ ! -d "$factor_dir" ]; then
        continue
    fi
    
    factor_name=$(basename "$factor_dir")
    
    # 跳过非因子目录
    if [ "$factor_name" = "all" ] || [ "$factor_name" = "all_plots_test" ]; then
        continue
    fi
    
    if [ -f "$factor_dir/README.md" ]; then
        echo "处理因子: $factor_name"
        
        # 提取关键信息
        if grep -q "🟢 alive" "$factor_dir/README.md"; then
            echo "- $factor_name" >> "$ALIVE_FILE"
            ALIVE_COUNT=$((ALIVE_COUNT + 1))
        elif grep -q "🟡 warning" "$factor_dir/README.md"; then
            echo "- $factor_name" >> "$WARNING_FILE"
            WARNING_COUNT=$((WARNING_COUNT + 1))
        else
            echo "- $factor_name" >> "$DEAD_FILE"
            DEAD_COUNT=$((DEAD_COUNT + 1))
        fi
        
        TOTAL_FACTORS=$((TOTAL_FACTORS + 1))
    fi
done

# 创建汇总文档
cat > "$OUTPUT_FILE" << EOF
# 因子测试结果汇总（格式化版）

生成时间: $(date '+%Y-%m-%d %H:%M:%S')

## 📊 快速概览

| 类别 | 数量 | 占比 |
|------|------|------|
| 🟢 Alive (优秀) | $ALIVE_COUNT | $(awk "BEGIN {printf \"%.1f%%\", $ALIVE_COUNT/$TOTAL_FACTORS*100}") |
| 🟡 Warning (注意) | $WARNING_COUNT | $(awk "BEGIN {printf \"%.1f%%\", $WARNING_COUNT/$TOTAL_FACTORS*100}") |
| 🔴 Dead (失效) | $DEAD_COUNT | $(awk "BEGIN {printf \"%.1f%%\", $DEAD_COUNT/$TOTAL_FACTORS*100}") |
| **总计** | **$TOTAL_FACTORS** | **100%** |

---

## 🟢 Alive 因子列表（推荐使用）

EOF

cat "$ALIVE_FILE" >> "$OUTPUT_FILE"

cat >> "$OUTPUT_FILE" << EOF

---

## 🟡 Warning 因子列表（需要监控）

EOF

cat "$WARNING_FILE" >> "$OUTPUT_FILE"

cat >> "$OUTPUT_FILE" << EOF

---

## 🔴 Dead 因子列表（不推荐）

EOF

cat "$DEAD_FILE" >> "$OUTPUT_FILE"

cat >> "$OUTPUT_FILE" << EOF

---

## 📈 详细对比

### 按调仓周期对比

以下表格展示了所有因子在不同调仓周期下的表现：

EOF

# 按周期汇总表格
for period in 5 10 15; do
    echo "" >> "$OUTPUT_FILE"
    echo "#### ${period}天周期" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "| 因子 | 等级 | 状态 | IC均值 | IR比率 | 多空年化 | 单调性 | 通过指标 |" >> "$OUTPUT_FILE"
    echo "|------|------|------|--------|--------|----------|--------|----------|" >> "$OUTPUT_FILE"
    
    # 提取每个因子的该周期数据
    for factor_dir in "$BATCH_DIR"/*/; do
        if [ ! -d "$factor_dir" ]; then
            continue
        fi
        
        factor_name=$(basename "$factor_dir")
        
        if [ "$factor_name" = "all" ] || [ "$factor_name" = "all_plots_test" ]; then
            continue
        fi
        
        if [ -f "$factor_dir/README.md" ]; then
            # 提取该周期的数据
            awk -v period="$period" '
                /^### '"$factor_name"' - [0-9]+天周期/ {
                    for (i=1; i<=50; i++) {
                        getline
                        if ($0 ~ /^IC均值.*\|/) {
                            ic = $3
                        }
                        if ($0 ~ /^IR比率.*\|/) {
                            ir = $3
                        }
                        if ($0 ~ /^多空年化.*\|/) {
                            ret = $3
                        }
                        if ($0 ~ /^单调性.*\|/) {
                            mono = $3
                        }
                        if ($0 ~ /^'"$factor_name"'.*天.*\|/) {
                            split($0, parts, "|")
                            grade = parts[3]
                            status = parts[4]
                            score = parts[5]
                            print parts[2] "|" grade "|" status "|" ic "|" ir "|" ret "|" mono "|" score
                            nextfile
                        }
                    }
                }
            ' "$factor_dir/README.md" >> "$OUTPUT_FILE" 2>/dev/null || true
        fi
    done
done

cat >> "$OUTPUT_FILE" << EOF

---

## 📂 查看详细结果

每个因子的完整测试报告位于：

\`\`\`
$BATCH_DIR/[因子名]/README.md
\`\`\`

---

## 📋 统计汇总

- **总因子数**: $TOTAL_FACTORS
- **Alive因子**: $ALIVE_COUNT ($(awk "BEGIN {printf \"%.1f%%\", $ALIVE_COUNT/$TOTAL_FACTORS*100}"))
- **Warning因子**: $WARNING_COUNT ($(awk "BEGIN {printf \"%.1f%%\", $WARNING_COUNT/$TOTAL_FACTORS*100}"))
- **Dead因子**: $DEAD_COUNT ($(awk "BEGIN {printf \"%.1f%%\", $DEAD_COUNT/$TOTAL_FACTORS*100}"))

生成时间: $(date '+%Y-%m-%d %H:%M:%S')

EOF

# 清理临时文件
rm -f "$SUMMARY_FILE" "$ALIVE_FILE" "$WARNING_FILE" "$DEAD_FILE"

echo ""
echo "✅ 格式化汇总完成！"
echo "📄 文件位置: $OUTPUT_FILE"
echo "📊 因子数: $TOTAL_FACTORS"
echo "  🟢 Alive: $ALIVE_COUNT"
echo "  🟡 Warning: $WARNING_COUNT"
echo "  🔴 Dead: $DEAD_COUNT"
echo ""
echo "查看汇总:"
echo "  cat $OUTPUT_FILE"
echo ""


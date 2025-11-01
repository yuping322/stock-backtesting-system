#!/usr/bin/env bash
# 总结 results/all/README.md 的结果

SOURCE_FILE="results/all/README.md"
OUTPUT_FILE="results/all/SUMMARY_FORMATTED.md"

echo "======================================================================"
echo "总结因子测试结果"
echo "======================================================================"
echo "源文件: $SOURCE_FILE"
echo "输出文件: $OUTPUT_FILE"
echo "======================================================================"

# 提取配置信息
START_DATE=$(grep "回测开始日期" "$SOURCE_FILE" | cut -d'|' -f3 | tr -d ' ')
END_DATE=$(grep "回测结束日期" "$SOURCE_FILE" | cut -d'|' -f3 | tr -d ' ')
STOCK_POOL=$(grep "股票池" "$SOURCE_FILE" | cut -d'|' -f3 | tr -d ' ')
FACTOR_COUNT=$(grep "因子数量" "$SOURCE_FILE" | cut -d'|' -f3 | tr -d ' ')

# 统计各类状态
ALIVE_COUNT=$(grep -c "🟢 alive" "$SOURCE_FILE" || echo "0")
WARNING_COUNT=$(grep -c "🟡 warning" "$SOURCE_FILE" || echo "0")
DEAD_COUNT=$(grep -c "🔴 dead" "$SOURCE_FILE" || echo "0")

# 计算总数
TOTAL_COUNT=$((ALIVE_COUNT + WARNING_COUNT + DEAD_COUNT))

echo "处理中: $TOTAL_COUNT 条记录"

# 创建汇总文件
cat > "$OUTPUT_FILE" << EOF
# 因子测试结果汇总

生成时间: $(date '+%Y-%m-%d %H:%M:%S')

## 📊 测试概览

| 参数 | 值 |
|------|-----|
| 回测开始日期 | $START_DATE |
| 回测结束日期 | $END_DATE |
| 股票池 | $STOCK_POOL |
| 因子数量 | $FACTOR_COUNT |
| 测试组合数 | $TOTAL_COUNT |

## 📈 统计结果

| 类别 | 数量 | 占比 |
|------|------|------|
| 🟢 Alive (优秀) | $ALIVE_COUNT | $(awk "BEGIN {printf \"%.1f%%\", $ALIVE_COUNT/$TOTAL_COUNT*100}") |
| 🟡 Warning (注意) | $WARNING_COUNT | $(awk "BEGIN {printf \"%.1f%%\", $WARNING_COUNT/$TOTAL_COUNT*100}") |
| 🔴 Dead (失效) | $DEAD_COUNT | $(awk "BEGIN {printf \"%.1f%%\", $DEAD_COUNT/$TOTAL_COUNT*100}") |
| **总计** | **$TOTAL_COUNT** | **100%** |

---

## 🟢 Alive 因子列表（推荐使用）

以下因子在所有周期测试中表现优秀：

| 因子 | 周期 | 等级 | 通过指标数 |
|------|------|------|-----------|
EOF

# 提取Alive因子
grep "🟢 alive" "$SOURCE_FILE" | awk -F'|' '{print "|", $2, "|", $3, "|", $4, "|", $6, "|"}' >> "$OUTPUT_FILE"

cat >> "$OUTPUT_FILE" << EOF

---

## 🟡 Warning 因子列表（需要监控）

以下因子表现一般，建议谨慎使用：

| 因子 | 周期 | 等级 | 通过指标数 |
|------|------|------|-----------|
EOF

# 提取Warning因子
grep "🟡 warning" "$SOURCE_FILE" | awk -F'|' '{print "|", $2, "|", $3, "|", $4, "|", $6, "|"}' >> "$OUTPUT_FILE"

cat >> "$OUTPUT_FILE" << EOF

---

## 🔴 Dead 因子列表（不推荐）

以下因子测试表现不佳，不推荐使用：

EOF

# 统计Dead因子的种类（去重）
echo "| 因子 | 测试周期数 | 平均通过指标 |" >> "$OUTPUT_FILE"
echo "|------|-----------|------------|" >> "$OUTPUT_FILE"

# 提取所有因子名称
awk -F'|' '/🟢 alive|🟡 warning|🔴 dead/ {print $2}' "$SOURCE_FILE" | sort | uniq | while read factor; do
    if [ -n "$factor" ]; then
        # 统计该因子的周期数
        cycle_count=$(grep "| $factor |" "$SOURCE_FILE" | wc -l | tr -d ' ')
        # 计算平均通过指标
        avg_score=$(grep "| $factor |" "$SOURCE_FILE" | awk -F'|' '{print $6}' | sed 's/\// /g' | awk '{sum1+=$1; sum2+=$2} END {if (sum2>0) printf "%.1f", sum1/sum2*9; else print "0"}' 2>/dev/null || echo "0")
        echo "| $factor | $cycle_count | $avg_score/9 |" >> "$OUTPUT_FILE"
    fi
done

cat >> "$OUTPUT_FILE" << EOF

---

## 📋 查看详细结果

完整结果请查看: [$SOURCE_FILE]($SOURCE_FILE)

---

## 📌 使用建议

1. **优先使用** 🟢 Alive 因子构建策略
2. **谨慎使用** 🟡 Warning 因子，需要定期监控
3. **避免使用** 🔴 Dead 因子

EOF

echo ""
echo "✅ 汇总完成！"
echo "📄 文件位置: $OUTPUT_FILE"
echo "📊 统计:"
echo "  🟢 Alive: $ALIVE_COUNT"
echo "  🟡 Warning: $WARNING_COUNT"
echo "  🔴 Dead: $DEAD_COUNT"
echo "  📊 总计: $TOTAL_COUNT"
echo ""
echo "查看汇总:"
echo "  cat $OUTPUT_FILE"
echo ""


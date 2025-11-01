#!/usr/bin/env bash
# 对比两个测试结果中的因子（改进版）

FILE1="results/all/README.md"          # 最近3个月
DIR2="results/factor_test_all_oss"     # 几年测试目录

OUTPUT_FILE="results/FACTOR_COMPARISON_V2.md"

echo "======================================================================"
echo "对比因子测试结果"
echo "======================================================================"
echo "文件1 (3个月): $FILE1"
echo "目录2 (几年): $DIR2"
echo "输出文件: $OUTPUT_FILE"
echo "======================================================================"

# 临时文件
TEMP1=$(mktemp)
TEMP2=$(mktemp)
COMMON=$(mktemp)
ONLY1=$(mktemp)
ONLY2=$(mktemp)

# 提取因子列表（去重）
echo "提取因子列表..."

# 从文件1提取因子
grep "^|" "$FILE1" | grep -E "🟢 alive|🟡 warning|🔴 dead" | awk -F'|' '{print $2}' | tr -d ' ' | sort | uniq > "$TEMP1"

# 从目录2提取因子（每个文件夹名就是一个因子）
ls "$DIR2" | grep -v README | grep -v ALL | grep -v "\.csv$" | sort | uniq > "$TEMP2"

# 统计数量
COUNT1=$(wc -l < "$TEMP1" | tr -d ' ')
COUNT2=$(wc -l < "$TEMP2" | tr -d ' ')

# 找出共同因子
comm -12 "$TEMP1" "$TEMP2" > "$COMMON"
COMMON_COUNT=$(wc -l < "$COMMON" | tr -d ' ')

# 找出只在文件1中的因子
comm -23 "$TEMP1" "$TEMP2" > "$ONLY1"
ONLY1_COUNT=$(wc -l < "$ONLY1" | tr -d ' ')

# 找出只在文件2中的因子
comm -13 "$TEMP1" "$TEMP2" > "$ONLY2"
ONLY2_COUNT=$(wc -l < "$ONLY2" | tr -d ' ')

echo "统计完成！"
echo "  文件1因子数: $COUNT1"
echo "  目录2因子数: $COUNT2"
echo "  共同因子数: $COMMON_COUNT"
echo "  仅文件1: $ONLY1_COUNT"
echo "  仅目录2: $ONLY2_COUNT"

# 生成对比报告
cat > "$OUTPUT_FILE" << EOF
# 因子测试结果对比

生成时间: $(date '+%Y-%m-%d %H:%M:%S')

## 📊 对比概览

| 项目 | 文件1 (3个月) | 目录2 (几年) | 说明 |
|------|--------------|-------------|------|
| 因子总数 | $COUNT1 | $COUNT2 | 去重后的因子数量 |
| 共同因子 | $COMMON_COUNT | $COMMON_COUNT | 在两个测试中都出现 |
| 仅文件1 | $ONLY1_COUNT | - | 只在3个月测试中出现 |
| 仅目录2 | - | $ONLY2_COUNT | 只在几年测试中出现 |

### 测试配置对比

**文件1 (results/all/README.md)**:
- 时间范围: 2025-07-26 ~ 2025-10-24 (3个月)
- 因子数: 252

**目录2 (results/factor_test_all_oss/)**:
- 时间范围: 2020-07-26 ~ 2025-10-24 (几年)
- 因子数: $COUNT2

---

## ✅ 共同因子列表 ($COMMON_COUNT 个)

以下因子在两个测试中都进行了测试：

EOF

# 添加共同因子列表
if [ $COMMON_COUNT -gt 0 ]; then
    echo "" >> "$OUTPUT_FILE"
    echo "| 序号 | 因子名称 |" >> "$OUTPUT_FILE"
    echo "|------|---------|" >> "$OUTPUT_FILE"
    nl -w1 -s' | ' "$COMMON" | sed 's/$/ |/' >> "$OUTPUT_FILE"
else
    echo "无共同因子" >> "$OUTPUT_FILE"
fi

cat >> "$OUTPUT_FILE" << EOF

---

## 📝 仅文件1中的因子 ($ONLY1_COUNT 个)

以下因子只在3个月测试中出现：

EOF

if [ $ONLY1_COUNT -gt 0 ]; then
    echo "" >> "$OUTPUT_FILE"
    echo "| 序号 | 因子名称 |" >> "$OUTPUT_FILE"
    echo "|------|---------|" >> "$OUTPUT_FILE"
    nl -w1 -s' | ' "$ONLY1" | sed 's/$/ |/' >> "$OUTPUT_FILE"
else
    echo "无" >> "$OUTPUT_FILE"
fi

cat >> "$OUTPUT_FILE" << EOF

---

## 📝 仅目录2中的因子 ($ONLY2_COUNT 个)

以下因子只在几年测试中出现：

EOF

if [ $ONLY2_COUNT -gt 0 ]; then
    echo "" >> "$OUTPUT_FILE"
    echo "| 序号 | 因子名称 |" >> "$OUTPUT_FILE"
    echo "|------|---------|" >> "$OUTPUT_FILE"
    nl -w1 -s' | ' "$ONLY2" | sed 's/$/ |/' >> "$OUTPUT_FILE"
else
    echo "无" >> "$OUTPUT_FILE"
fi

cat >> "$OUTPUT_FILE" << EOF

---

## 📌 重复因子分析

### 重复率

- **文件1在目录2中的覆盖率**: $(awk "BEGIN {printf \"%.1f%%\", $COMMON_COUNT/$COUNT1*100}")
- **目录2在文件1中的覆盖率**: $(awk "BEGIN {printf \"%.1f%%\", $COMMON_COUNT/$COUNT2*100}")

### 说明

- **共同因子**: 在两个测试中都出现的因子，可以用于对比时间段的稳定性
- **仅文件1**: 可能在最近才添加到测试列表中的因子
- **仅目录2**: 可能在历史测试中使用的因子

---

## 🔍 查看详细结果

- 文件1 (3个月): [results/all/README.md](results/all/README.md)
- 目录2 (几年): [results/factor_test_all_oss/](results/factor_test_all_oss/)

生成时间: $(date '+%Y-%m-%d %H:%M:%S')

EOF

# 清理临时文件
rm -f "$TEMP1" "$TEMP2" "$COMMON" "$ONLY1" "$ONLY2"

echo ""
echo "✅ 对比完成！"
echo "📄 文件位置: $OUTPUT_FILE"
echo ""
echo "查看对比:"
echo "  cat $OUTPUT_FILE"
echo ""


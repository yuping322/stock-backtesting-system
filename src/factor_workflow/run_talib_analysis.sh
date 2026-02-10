#!/bin/bash
# 基于 TALIB 因子测试结果的使用示例
# 使用筛选出的优秀因子进行进一步分析

echo "=== TALIB 优秀因子后续分析示例 ==="
echo

# 检查优秀因子文件是否存在
if [ ! -f "../results/good_talib_factors.txt" ]; then
    echo "错误：找不到优秀因子列表文件 ../results/good_talib_factors.txt"
    echo "请先运行: python ../talib_analysis_workflow.py"
    exit 1
fi

# 读取优秀因子列表
GOOD_FACTORS=$(cat ../results/good_talib_factors.txt | tr '\n' ' ')
echo "优秀因子: $GOOD_FACTORS"
echo

echo "=== 步骤1：生成 formatted_data.csv ==="
echo "python ../factor/export_data.py \\"
echo "    --stocks 000001 000002 000858 600036 \\"
echo "    --factors $GOOD_FACTORS \\"
echo "    --mode custom \\"
echo "    --output ../data/model_tasks"
echo

# 实际执行命令
python ../factor/export_data.py \
    --stocks 000001 000002 000858 600036 \
    --factors $GOOD_FACTORS \
    --mode custom \
    --output ../data/model_tasks

if [ $? -eq 0 ]; then
    echo "✓ formatted_data.csv 生成完成"
else
    echo "✗ 数据导出失败"
    exit 1
fi

echo
echo "=== 步骤2：转换为 factor_workflow 格式 ==="
echo "python convert_sample.py \\"
echo "    --input-csv ../data/model_tasks/formatted_data.csv \\"
echo "    --limit-stocks 50"

python convert_sample.py \
    --input-csv ../data/model_tasks/formatted_data.csv \
    --limit-stocks 50

if [ $? -eq 0 ]; then
    echo "✓ 数据转换完成"
else
    echo "✗ 数据转换失败"
    exit 1
fi

echo
echo "=== 步骤3：运行 factor_workflow 生成预测权重 ==="
echo "python workflow_main.py && \\"
echo "python backtest_evaluation.py && \\"
echo "python export_scores.py"

python workflow_main.py
if [ $? -ne 0 ]; then
    echo "✗ 模型训练失败"
    exit 1
fi

python backtest_evaluation.py
if [ $? -ne 0 ]; then
    echo "✗ 回测评估失败"
    exit 1
fi

python export_scores.py
if [ $? -ne 0 ]; then
    echo "✗ 权重导出失败"
    exit 1
fi

echo "✓ factor_workflow 完成"

echo
echo "=== 分析完成！==="
echo
echo "输出文件位置："
echo "- 优秀因子列表: ../results/good_talib_factors.txt"
echo "- 宽表数据: ../data/model_tasks/formatted_data.csv"
echo "- 因子数据: ../data/model_tasks/features_panel.pkl"
echo "- 标签数据: ../data/model_tasks/label_panel.pkl"
echo "- 元数据: ../data/model_tasks/meta_series.pkl"
echo "- 预测权重CSV: ../results/factor_workflow/scores.csv"
echo "- 回测结果: ../results/factor_workflow/backtest/"
echo
echo "现在你可以使用 ../results/factor_workflow/scores.csv 中的权重"
echo "来进行实际的股票投资组合构建和回测分析。"
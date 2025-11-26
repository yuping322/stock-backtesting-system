#!/bin/bash
# TALIB 因子深入分析示例脚本
# 基于 factor_test_talib_202509 结果，使用优秀因子进行进一步分析

echo "=== TALIB 因子深入分析示例 ==="
echo

# 读取优秀因子列表
if [ ! -f "results/good_talib_factors.txt" ]; then
    echo "错误：找不到优秀因子列表文件 results/good_talib_factors.txt"
    echo "请先运行分析脚本筛选优秀因子"
    exit 1
fi

# 读取因子列表
GOOD_FACTORS=$(cat results/good_talib_factors.txt | tr '\n' ',' | sed 's/,$//')
echo "优秀因子列表: $GOOD_FACTORS"
echo

# 示例1：使用优秀因子进行详细测试（生成图表）
echo "=== 示例1：优秀因子详细测试（带图表）==="
echo "python main_factor.py \\
    --start 2025-09-01 \\
    --end 2025-11-23 \\
    --factors $GOOD_FACTORS \\
    --plot true \\
    --plot-mode save \\
    --output-dir results/talib_good_factors_detailed \\
    --quantiles 10 \\
    --periods 5,10,20 \\
    --roll-win 60"
echo

# 示例2：针对特定因子进行参数调优
echo "=== 示例2：MACD 因子参数调优测试 ==="
echo "python main_factor.py \\
    --start 2025-09-01 \\
    --end 2025-11-23 \\
    --factors TALIB_MACD_12_26_9,TALIB_MACDEXT_12_26_9_0_0_0,TALIB_MACDFIX_9 \\
    --plot true \\
    --plot-mode save \\
    --output-dir results/talib_macd_tuning \\
    --quantiles 5 \\
    --periods 1,3,5,10 \\
    --roll-win 30"
echo

# 示例3：使用因子目录模式（如果有原始因子数据）
echo "=== 示例3：使用因子目录模式 ==="
echo "# 如果你有原始因子数据CSV文件，可以使用因子目录模式"
echo "# 假设因子文件在 data/talib_factors/ 目录下"
echo "python main_factor.py \\
    --factor-dir data/talib_factors \\
    --start 2025-09-01 \\
    --end 2025-11-23 \\
    --plot true \\
    --plot-mode popup \\
    --output-dir results/talib_from_files \\
    --quantiles 10 \\
    --periods 5,10,15"
echo

# 示例4：转换为 factor_workflow 格式
echo "=== 示例4：转换为 factor_workflow 格式 ==="
echo "# 将因子数据转换为 QLib panel 格式，用于机器学习训练"
echo "python convert_factor_to_workflow.py \\
    --factor-csv results/talib_good_factors_detailed \\
    --output-dir exported_data_all"
echo

# 示例5：运行完整的 factor_workflow 流程
echo "=== 示例5：运行 factor_workflow 机器学习流程 ==="
echo "cd factor_workflow"
echo "python workflow_main.py    # 训练模型"
echo "python backtest_evaluation.py  # 评估和回测"
echo "python export_scores.py    # 导出预测权重"
echo

echo "=== 运行建议 ==="
echo "1. 从示例1开始，验证优秀因子的表现"
echo "2. 根据需要调整参数（分位数、周期等）"
echo "3. 如果有原始因子数据，使用示例3的因子目录模式"
echo "4. 要生成可交易权重，使用示例4和5转换为ML流程"
echo

echo "=== 输出文件说明 ==="
echo "- results/talib_good_factors_detailed/: 详细测试报告和图表"
echo "- results/factor_workflow/scores.csv: 最终预测权重"
echo "- results/factor_workflow/backtest/: 回测结果"
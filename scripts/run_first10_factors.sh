#!/bin/bash
# 测试前10个因子（从 OSS 读取）

python main_factor.py \
  --factors PLRC24 operating_cost_ttm BIAS10 cash_to_current_liability equity_to_fixed_asset_ratio VEMA12 PEG bull_power Price1Y AR \
  --start 2025-07-26 \
  --end 2025-10-24 \
  --output-dir results/factor_test_first10 

echo ""
echo "✅ 测试完成！"
echo "📊 查看报告: cat results/factor_test_first10/README.md"
echo "🖼️  查看图片: ls results/factor_test_first10/*/plot_*.png"

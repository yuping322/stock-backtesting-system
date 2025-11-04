#!/bin/bash
# 测试Alpha158因子文件中的前10个因子（使用main_factor.py）

python main_factor.py \
  --factors KMID KLEN KMID2 KUP KUP2 KLOW KLOW2 KSFT KSFT2 OPEN0 \
  --start 2025-08-09 \
  --end 2025-10-31 \
  --factor-dir factors \
  --output-dir results/alpha158_first10 

echo ""
echo "✅ 测试完成！"
echo "📊 查看报告: cat results/alpha158_first10/README.md"
echo "📊 查看监控: cat results/alpha158_first10/monitor.csv"
echo "🖼️  查看图片: ls results/alpha158_first10/*/plot_*.png"

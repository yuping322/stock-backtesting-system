#!/bin/bash
# 快速测试脚本

echo "================================"
echo "运行ML训练脚本"
echo "================================"
echo ""

# 激活虚拟环境
source .venv/bin/activate

# 运行训练脚本
python ml/train_super_factor_v2.py

echo ""
echo "完成！检查输出："
echo "  - ml/models/ - 训练好的模型"
echo "  - ml/output/model_comparison.csv - 模型对比结果"


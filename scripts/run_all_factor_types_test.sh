#!/bin/bash
# 运行覆盖所有因子类型的因子检验测试

cd "$(dirname "$0")/.."

echo "============================================================"
echo "运行覆盖所有因子类型的因子检验测试"
echo "============================================================"
echo ""
echo "测试包括以下因子类型："
echo "  1. 内置函数因子: VOL10, single_day_VPT_12"
echo "  2. OSS因子: OPEN0, HIGH0 (尝试从OSS读取)"
echo "  3. Alpha文件因子: KMID, KLEN (从Alpha158文件加载)"
echo "  4. 自定义因子: volume_ma20, price_momentum (临时生成)"
echo ""
echo "结果将保存到: results/test_all_factor_types/"
echo ""

# 运行测试
python scripts/test_all_factor_types.py

echo ""
echo "============================================================"
echo "测试完成！"
echo "============================================================"
echo ""
echo "查看结果:"
echo "  - 汇总文件: results/test_all_factor_types/monitor.csv"
echo "  - 详细结果: results/test_all_factor_types/<因子名>/README_<周期>.md"
echo ""


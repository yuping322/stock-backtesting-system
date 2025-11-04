#!/bin/bash
# 测试Alpha158因子文件中的前10个因子

# 因子文件路径
FACTOR_FILE="factors_test/Alpha158_20251004_20251103.csv"
FACTOR_DIR="factors_test"

echo "============================================================"
echo "测试Alpha158因子文件中的前10个因子"
echo "============================================================"
echo ""

# 使用Python脚本运行测试
python << 'PYTHON_SCRIPT'
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent if '__file__' in globals() else Path.cwd()
sys.path.insert(0, str(project_root))

import pandas as pd
from factor.factor import FactorTester, CFG
from factor.factor_calculator import create_factor_calculator
from datetime import datetime

# 因子文件路径
FACTOR_FILE = "factors_test/Alpha158_20251004_20251103.csv"
FACTOR_DIR = "factors_test"
OUTPUT_DIR = "results/alpha158_test_first10"

# 1. 读取因子文件，提取因子列表和日期范围
print("📊 读取因子文件...")
df = pd.read_csv(FACTOR_FILE)

# 提取因子列表（前10个）
factors = [c for c in df.columns if c not in ['date', 'code']]
test_factors = factors[:10]
print(f"✅ 测试因子: {', '.join(test_factors)}")

# 提取日期范围
start_date = df['date'].min()
end_date = df['date'].max()
print(f"✅ 日期范围: {start_date} 到 {end_date}")

# 提取股票代码（使用因子文件中已有的股票）
codes = sorted(df['code'].unique().astype(str))
# 确保代码格式正确（6位数字）
codes = [str(c).zfill(6) for c in codes]
# 限制前20只股票（加快测试速度）
test_codes = codes[:20] if len(codes) > 20 else codes
print(f"✅ 使用股票: {', '.join(test_codes[:5])}... (共{len(test_codes)}只)")
print()

# 2. 创建配置
class Args:
    start = start_date
    end = end_date
    stock_pool = '000300'  # 用于获取行业信息，但实际使用因子文件中的股票
    factors = test_factors
    quantiles = 5
    periods = [5, 10]
    fillna = 0
    winsorize = 0
    neutralize = 0
    standardize = 0
    roll_win = 20
    monitor_csv = f"{OUTPUT_DIR}/monitor.csv"
    last_only = False
    max_stocks = None
    factor_dir = FACTOR_DIR

args = Args()
cfg = CFG(args)

# 3. 创建因子计算器
print("🔧 创建因子计算器...")
custom_factors = {}
for factor_name in test_factors:
    try:
        calc = create_factor_calculator(factor_name=factor_name, factor_dir=FACTOR_DIR)
        custom_factors[factor_name] = calc
        print(f"  ✓ {factor_name}")
    except Exception as e:
        print(f"  ✗ {factor_name}: {e}")

if not custom_factors:
    print("❌ 未成功创建任何因子计算器")
    sys.exit(1)

print()

# 4. 创建测试器
print("🚀 开始测试因子...")
print("=" * 80)
tester = FactorTester(cfg, custom_factors=custom_factors)

# 直接设置股票列表（使用因子文件中的股票）
# 注意：需要在get_stocks()之前设置，并在get_stocks()之后恢复
print(f"使用股票列表: {len(test_codes)} 只股票")

# 保存原始get_stocks方法
original_get_stocks = tester.get_stocks

# 重写get_stocks方法，直接使用因子文件中的股票
def custom_get_stocks():
    tester.stocks = test_codes
    print(f"[自定义] 使用因子文件中的股票: {len(test_codes)} 只")

tester.get_stocks = custom_get_stocks

# 5. 运行测试
try:
    results = tester.run(plot=False)
    print()
    print("=" * 80)
    print(f"✅ 测试完成！验证了 {len(results)} 个因子-周期组合")
    print("=" * 80)
    print()
    print(f"📊 监控文件: {OUTPUT_DIR}/monitor.csv")
    print()
    
    # 显示监控结果摘要
    if os.path.exists(f"{OUTPUT_DIR}/monitor.csv"):
        monitor_df = pd.read_csv(f"{OUTPUT_DIR}/monitor.csv")
        print("📈 因子表现摘要:")
        print("-" * 80)
        for factor in test_factors:
            factor_data = monitor_df[monitor_df['factor'] == factor]
            if not factor_data.empty:
                best_period = factor_data.loc[factor_data['roll_ic'].abs().idxmax()]
                ir_val = best_period['roll_ir']
                ir_str = f"{ir_val:6.3f}" if pd.notna(ir_val) else "   N/A"
                print(f"  {factor:10s} - 周期{int(best_period['period']):2d}天: IC={best_period['roll_ic']:6.3f}, IR={ir_str}")
        print()
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

PYTHON_SCRIPT

echo ""
echo "============================================================"
echo "✅ 测试完成！"
echo "============================================================"
echo ""
echo "📊 查看监控结果:"
echo "  cat results/alpha158_test_first10/monitor.csv"
echo ""

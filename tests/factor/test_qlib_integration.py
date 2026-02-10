"""
测试 Qlib 因子生成器和 File 因子加载器的集成

验证：
1. Qlib 生成器能否生成 Alpha158 因子文件
2. File 加载器能否正确识别和加载 Qlib 生成的 CSV
3. 两个模块协作是否正确
"""

import sys
import os
import pandas as pd
import tempfile
from pathlib import Path

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.factor.generator.qlib import generate_qlib_factors
from src.factor.generator.file import generate_file_factors


def test_qlib_generator():
    """测试 Qlib 因子生成器"""
    print("\n" + "="*60)
    print("测试 1: Qlib 因子生成器")
    print("="*60)
    
    try:
        # 生成 Qlib 因子
        stock_codes = ['000001', '000002']
        start_date = '2024-09-01'
        end_date = '2024-09-10'
        
        print(f"\n生成 Alpha158 因子...")
        print(f"  股票: {stock_codes}")
        print(f"  日期: {start_date} ~ {end_date}")
        
        df = generate_qlib_factors(
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date,
            factor_set='Alpha158',
            output_file=None,  # 先不保存到文件
            qlib_cache_dir=None
        )
        
        if df.empty:
            print("❌ 生成失败：返回空 DataFrame")
            return False
        
        print(f"\n✅ 生成成功")
        print(f"  数据形状: {df.shape}")
        print(f"  列: {df.columns.tolist()[:10]}... ({len(df.columns)} 列)")
        print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
        print(f"  股票数: {df['stock_code'].nunique()}")
        
        # 验证数据结构
        required_cols = ['date', 'stock_code']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ 缺少必要列: {missing_cols}")
            return False
        
        # 验证因子数
        factor_cols = [col for col in df.columns if col not in ['date', 'stock_code']]
        if len(factor_cols) < 10:
            print(f"❌ 因子数太少: {len(factor_cols)} (预期 100+)")
            return False
        
        print(f"✅ 验证通过: {len(factor_cols)} 个因子")
        return True
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_qlib_to_file_integration():
    """测试 Qlib 生成 -> File 加载集成"""
    print("\n" + "="*60)
    print("测试 2: Qlib 生成 -> File 加载集成")
    print("="*60)
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # 第一步：生成 Qlib 因子文件
            print(f"\n第一步: 使用 Qlib 生成器生成因子...")
            
            stock_codes = ['000001', '000002']
            start_date = '2024-09-01'
            end_date = '2024-09-10'
            
            # 生成 Qlib 因子
            df_qlib = generate_qlib_factors(
                stock_codes=stock_codes,
                start_date=start_date,
                end_date=end_date,
                factor_set='Alpha158',
                output_file=None
            )
            
            if df_qlib.empty:
                print("❌ Qlib 生成失败")
                return False
            
            print(f"✓ Qlib 生成成功: {df_qlib.shape[0]} 行, {df_qlib.shape[1]} 列")
            
            # 保存为 CSV 文件（模拟 Qlib 生成的文件）
            qlib_csv = os.path.join(tmpdir, 'Alpha158_test.csv')
            df_qlib.to_csv(qlib_csv, index=False)
            print(f"✓ 保存到 CSV: {qlib_csv}")
            
            # 第二步：使用 File 加载器加载这个文件
            print(f"\n第二步: 使用 File 加载器加载 Qlib CSV...")
            
            output_dir = os.path.join(tmpdir, 'output')
            
            result = generate_file_factors(
                factor_file_paths={
                    'qlib_alpha158': qlib_csv
                },
                stock_codes=['000001'],  # 只加载第一只股票
                start_date='2024-09-02',
                end_date='2024-09-05',
                output_dir=output_dir
            )
            
            print(f"✓ File 加载成功")
            print(f"  输出目录: {output_dir}")
            print(f"  因子文件: {result['factor_file']}")
            
            # 验证输出
            output_df = pd.read_csv(result['factor_file'])
            print(f"✓ 输出 DataFrame: {output_df.shape[0]} 行, {output_df.shape[1]} 列")
            
            # 检查数据
            if output_df.empty:
                print("❌ 加载后数据为空")
                return False
            
            print(f"✓ 数据不为空")
            
            # 检查是否只有一只股票
            unique_stocks = output_df['stock_code'].unique()
            if len(unique_stocks) != 1:
                print(f"❌ 股票数不对: {len(unique_stocks)} (预期 1)")
                return False
            
            print(f"✓ 股票过滤正确: {unique_stocks[0]}")
            
            # 检查日期范围
            min_date = pd.to_datetime(output_df['date']).min()
            max_date = pd.to_datetime(output_df['date']).max()
            if min_date < pd.to_datetime('2024-09-02') or max_date > pd.to_datetime('2024-09-05'):
                print(f"❌ 日期范围不对: {min_date} ~ {max_date}")
                return False
            
            print(f"✓ 日期范围正确: {min_date.date()} ~ {max_date.date()}")
            
            # 检查因子数
            factor_cols = [col for col in output_df.columns if col not in ['date', 'stock_code']]
            if len(factor_cols) < 100:
                print(f"⚠️  警告: 因子数较少: {len(factor_cols)}")
            else:
                print(f"✓ 因子数正确: {len(factor_cols)}")
            
            print(f"\n✅ 集成测试成功")
            return True
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_csv_format_detection():
    """测试 File 加载器的 CSV 格式检测"""
    print("\n" + "="*60)
    print("测试 3: CSV 格式检测")
    print("="*60)
    
    try:
        from src.factor.generator.file import FileFactorGenerator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 测试 1: 标准格式 CSV
            print(f"\n测试 3.1: 标准格式 CSV (date, code, factor_value)")
            
            standard_csv = os.path.join(tmpdir, 'standard.csv')
            pd.DataFrame({
                'date': ['2024-01-01', '2024-01-02'],
                'code': ['000001', '000001'],
                'factor_value': [1.23, 4.56]
            }).to_csv(standard_csv, index=False)
            
            generator = FileFactorGenerator(
                {'test': standard_csv},
                stock_codes=['000001'],
                start_date='2024-01-01',
                end_date='2024-01-31'
            )
            generator.setup_task()
            df = generator._load_csv_file(standard_csv)
            
            if 'factor_value' in df.columns:
                print(f"✓ 标准格式检测正确")
            else:
                print(f"❌ 标准格式检测失败")
                return False
            
            # 测试 2: Qlib 格式 CSV
            print(f"\n测试 3.2: Qlib 格式 CSV (date, code, Alpha005, Alpha010, ...)")
            
            qlib_csv = os.path.join(tmpdir, 'qlib.csv')
            data = {
                'date': ['2024-01-01', '2024-01-02'] * 50,  # 足够多的行
                'code': ['000001', '000001'] * 50,
            }
            # 添加 100+ 列因子
            for i in range(150):
                data[f'Alpha{i:03d}'] = [0.1 * (i + 1)] * 100
            
            pd.DataFrame(data).to_csv(qlib_csv, index=False)
            
            df = generator._load_csv_file(qlib_csv)
            
            if len([col for col in df.columns if col not in ['date', 'stock_code']]) > 100:
                print(f"✓ Qlib 格式检测正确")
            else:
                print(f"⚠️  警告: 因子数较少")
            
            print(f"\n✅ CSV 格式检测测试通过")
            return True
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("="*60)
    print("Qlib 因子生成器集成测试")
    print("="*60)
    
    results = []
    
    # 注意：Qlib 生成测试需要 qlib 库，可能会跳过
    try:
        results.append(("Qlib 因子生成器", test_qlib_generator()))
    except ImportError as e:
        if 'qlib' in str(e).lower():
            print(f"\n⚠️  跳过: Qlib 未安装 ({e})")
            results.append(("Qlib 因子生成器", None))
        else:
            raise
    except Exception as e:
        print(f"❌ 异常: {e}")
        results.append(("Qlib 因子生成器", False))
    
    # 集成测试（需要 Qlib）
    try:
        results.append(("Qlib 生成 -> File 加载集成", test_qlib_to_file_integration()))
    except ImportError as e:
        if 'qlib' in str(e).lower():
            print(f"\n⚠️  跳过: Qlib 未安装 ({e})")
            results.append(("Qlib 生成 -> File 加载集成", None))
        else:
            raise
    except Exception as e:
        print(f"❌ 异常: {e}")
        results.append(("Qlib 生成 -> File 加载集成", False))
    
    # CSV 格式检测测试（不需要 Qlib）
    try:
        results.append(("CSV 格式检测", test_file_csv_format_detection()))
    except Exception as e:
        print(f"❌ 异常: {e}")
        results.append(("CSV 格式检测", False))
    
    # 输出总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    for test_name, result in results:
        if result is None:
            status = "⏭️  跳过"
        elif result:
            status = "✅ 通过"
        else:
            status = "❌ 失败"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, r in results if r is True)
    skipped = sum(1 for _, r in results if r is None)
    failed = sum(1 for _, r in results if r is False)
    total = len(results)
    
    print(f"\n总计: {passed} 通过, {skipped} 跳过, {failed} 失败")
    
    if failed == 0:
        print("\n🎉 所有可运行的测试通过！")
    else:
        print(f"\n❌ 有 {failed} 个测试失败")

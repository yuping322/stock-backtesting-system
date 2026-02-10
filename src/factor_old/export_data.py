#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出价格数据和因子数据
用于在其他平台进行建模和测试

用法:
    python export_data.py --stocks 000001 000002 600000 --factors VOL10 VSTD10 --output ./exported_data
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd
import numpy as np
"""
导出完整的格式化数据：基础行情+原生OSS因子+可选TALIB因子
date, stock, open, high, low, close, volume, amount, mkt_cap, industry, concepts, <因子字段>

Args:
    codes: 股票代码列表
    start_date: 开始日期 'YYYY-MM-DD'
    end_date: 结束日期 'YYYY-MM-DD'
    output_dir: 输出目录
    factors: OSS因子列表，None或[]时为全量
    industry_default: 默认行业分类
    include_talib: 是否包含TALIB因子（从文件加载或直接计算）
    talib_factors: TALIB因子列表，None时使用默认列表
    compute_talib: 是否直接计算TALIB因子（不从文件加载）

Returns:
    str: 输出文件路径
"""
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import data


def get_last_3_months() -> tuple[str, str]:
    """
    获取最近3个月的日期范围
    
    Returns:
        tuple: (start_date, end_date) 格式为 'YYYY-MM-DD'
    """
    end_date = datetime.now().date()
    # 往前推3个月（约90天）
    start_date = end_date - timedelta(days=90)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def export_price_data(
    codes: List[str],
    start_date: str,
    end_date: str,
    output_dir: str
) -> str:
    """
    导出价格数据（OHLCV）
    
    Args:
        codes: 股票代码列表
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        output_dir: 输出目录
        
    Returns:
        str: 输出文件路径
    """
    print(f"\n{'='*60}")
    print(f"正在导出价格数据...")
    print(f"股票数量: {len(codes)}")
    print(f"日期范围: {start_date} ~ {end_date}")
    print(f"{'='*60}\n")
    
    # 获取所有字段的价格数据
    price_dict = data.load_oss_complex_stocks(
        codes=codes,
        start=start_date,
        end=end_date,
        fields='all'
    )
    
    if not price_dict:
        print("⚠️  未获取到任何价格数据")
        return None
    
    # 合并所有字段到一个 DataFrame
    price_frames = []
    
    for field_name, field_df in price_dict.items():
        # 转成长格式
        field_long = field_df.reset_index().melt(
            id_vars='date',
            var_name='code',
            value_name=field_name
        )
        
        if len(price_frames) == 0:
            price_frames.append(field_long)
        else:
            # 合并到第一个 DataFrame
            price_frames[0] = price_frames[0].merge(
                field_long,
                on=['date', 'code'],
                how='outer'
            )
    
    price_df = price_frames[0] if price_frames else pd.DataFrame()
    
    if price_df.empty:
        print("⚠️  价格数据为空")
        return None
    
    # 确保列顺序：date, code, open, high, low, close, volume, ...
    preferred_order = ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']
    other_cols = [c for c in price_df.columns if c not in preferred_order]
    column_order = [c for c in preferred_order if c in price_df.columns] + other_cols
    price_df = price_df[column_order]
    
    # 按日期和代码排序
    price_df = price_df.sort_values(['date', 'code']).reset_index(drop=True)
    
    # 保存到 CSV
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'price_data.csv')
    price_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"✓ 价格数据已导出: {output_file}")
    print(f"  数据形状: {price_df.shape}")
    print(f"  日期范围: {price_df['date'].min()} ~ {price_df['date'].max()}")
    print(f"  股票数量: {price_df['code'].nunique()}")
    
    return output_file


def export_factor_data(
    codes: List[str],
    factors: List[str],
    start_date: str,
    end_date: str,
    output_dir: str
) -> str:
    """
    导出因子数据
    
    Args:
        codes: 股票代码列表
        factors: 因子名称列表
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        output_dir: 输出目录
        
    Returns:
        str: 输出文件路径
    """
    print(f"\n{'='*60}")
    print(f"正在导出因子数据...")
    print(f"股票数量: {len(codes)}")
    print(f"因子列表: {factors}")
    print(f"日期范围: {start_date} ~ {end_date}")
    print(f"{'='*60}\n")
    
    factor_dict = {}
    
    for i, factor_name in enumerate(factors, 1):
        print(f"[{i}/{len(factors)}] 导出因子: {factor_name}")
        
        try:
            # 获取因子数据
            factor_series = data.factor_for_al(
                codes=codes,
                start_date=start_date,
                end_date=end_date,
                factor_name=factor_name
            )
            
            if factor_series.empty:
                print(f"  ⚠️  因子 {factor_name} 无数据，跳过")
                continue
            
            # 转换为 DataFrame
            factor_df = factor_series.reset_index()
            factor_df.columns = ['date', 'code', factor_name]
            
            # 存储
            factor_dict[factor_name] = factor_df
            
            print(f"  ✓ 因子 {factor_name} 数据点: {len(factor_df)}")
            
        except Exception as e:
            print(f"  ❌ 因子 {factor_name} 导出失败: {e}")
            continue
    
    if not factor_dict:
        print("⚠️  未导出任何因子数据")
        return None
    
    # 合并所有因子
    factor_frames = list(factor_dict.values())
    factor_df = factor_frames[0]
    
    for i in range(1, len(factor_frames)):
        factor_df = factor_df.merge(
            factor_frames[i],
            on=['date', 'code'],
            how='outer'
        )
    
    # 按日期和代码排序
    factor_df = factor_df.sort_values(['date', 'code']).reset_index(drop=True)
    
    # 保存到 CSV
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'factor_data.csv')
    factor_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n✓ 因子数据已导出: {output_file}")
    print(f"  数据形状: {factor_df.shape}")
    print(f"  日期范围: {factor_df['date'].min()} ~ {factor_df['date'].max()}")
    print(f"  股票数量: {factor_df['code'].nunique()}")
    print(f"  因子数量: {len(factor_df.columns) - 2}")  # 减去 date 和 code
    
    return output_file


def export_combined_data(
    codes: List[str],
    factors: List[str],
    start_date: str,
    end_date: str,
    output_dir: str
) -> str:
    """
    导出合并的价格和因子数据（宽表格式）
    
    Args:
        codes: 股票代码列表
        factors: 因子名称列表
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        output_dir: 输出目录
        
    Returns:
        str: 输出文件路径
    """
    print(f"\n{'='*60}")
    print(f"正在导出合并数据...")
    print(f"{'='*60}\n")
    
    # 加载价格数据
    price_dict = data.load_oss_complex_stocks(
        codes=codes,
        start=start_date,
        end=end_date,
        fields='all'
    )
    
    # 加载因子数据
    factor_dict = {}
    for factor_name in factors:
        try:
            factor_series = data.factor_for_al(
                codes=codes,
                start_date=start_date,
                end_date=end_date,
                factor_name=factor_name
            )
            if not factor_series.empty:
                factor_dict[factor_name] = factor_series
        except Exception as e:
            print(f"⚠️  因子 {factor_name} 加载失败: {e}")
    
    # 构建合并数据
    all_data = []
    
    # 获取所有日期和代码的组合
    if price_dict:
        first_price_field = list(price_dict.values())[0]
        date_code_combinations = [
            (date, code) 
            for date in first_price_field.index 
            for code in first_price_field.columns
        ]
    elif factor_dict:
        first_factor = list(factor_dict.values())[0]
        date_code_combinations = list(first_factor.index)
    else:
        print("⚠️  无任何数据可合并")
        return None
    
    # 为每个 (date, code) 组合收集数据
    for date, code in date_code_combinations:
        row = {'date': date, 'code': code}
        
        # 添加价格数据
        for field_name, field_df in price_dict.items():
            try:
                if date in field_df.index and code in field_df.columns:
                    row[field_name] = field_df.loc[date, code]
            except:
                pass
        
        # 添加因子数据
        for factor_name, factor_series in factor_dict.items():
            try:
                if (date, code) in factor_series.index:
                    row[factor_name] = factor_series.loc[(date, code)]
            except:
                pass
        
        all_data.append(row)
    
    combined_df = pd.DataFrame(all_data)
    
    if combined_df.empty:
        print("⚠️  合并数据为空")
        return None
    
    # 确保列顺序
    preferred_order = ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']
    other_cols = [c for c in combined_df.columns if c not in preferred_order]
    column_order = [c for c in preferred_order if c in combined_df.columns] + sorted(other_cols)
    combined_df = combined_df[column_order]
    
    # 按日期和代码排序
    combined_df = combined_df.sort_values(['date', 'code']).reset_index(drop=True)
    
    # 保存到 CSV
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'combined_data.csv')
    combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"✓ 合并数据已导出: {output_file}")
    print(f"  数据形状: {combined_df.shape}")
    print(f"  日期范围: {combined_df['date'].min()} ~ {combined_df['date'].max()}")
    print(f"  股票数量: {combined_df['code'].nunique()}")
    print(f"  价格字段: {len([c for c in combined_df.columns if c in ['open', 'high', 'low', 'close', 'volume']])}")
    print(f"  因子字段: {len([c for c in combined_df.columns if c not in ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']])}")
    
    return output_file


def export_formatted_csv(
    codes: List[str],
    start_date: str,
    end_date: str,
    output_dir: str,
    factors: List[str] = None,   # 允许指定因子子集，None或[]时为全量
    industry_default: str = "Unknown",
    include_talib: bool = False,  # 是否包含TALIB因子（从文件加载或直接计算）
    talib_factors: List[str] = None,  # TALIB因子列表
    compute_talib: bool = False  # 是否直接计算TALIB因子（不从文件加载）
) -> str:
    """
    导出完整的格式化数据：基础行情+原生OSS因子+可选TALIB因子
    date, stock, open, high, low, close, volume, amount, mkt_cap, industry, concepts, <因子字段>

    Args:
        codes: 股票代码列表
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        output_dir: 输出目录
        factors: OSS因子列表，None或[]时为全量
        industry_default: 默认行业分类
        include_talib: 是否包含TALIB技术指标因子
        talib_factors: TALIB因子列表，None时使用默认列表

    Returns:
        str: 输出文件路径
    """
    print(f"\n{'='*60}")
    print(f"导出完整格式化数据...")
    print(f"股票数量: {len(codes)}  日期: {start_date} ~ {end_date}")
    print(f"OSS因子: {'ALL' if not factors else factors}")
    print(f"TALIB因子: {'YES' if include_talib else 'NO'}")
    if include_talib and talib_factors:
        print(f"TALIB列表: {talib_factors}")
    print(f"{'='*60}\n")

    # --------- 1. 加载行情数据 ---------
    price_dict = data.load_oss_complex_stocks(
        codes=codes,
        start=start_date,
        end=end_date,
        fields="all"
    )
    if not price_dict:
        print("⚠️  未读取到任何行情数据")
        return None

    # 合并行情长表
    merged = None
    for fname, fdf in price_dict.items():
        long_df = fdf.reset_index().melt(
            id_vars='date', var_name='stock', value_name=fname
        )
        if merged is None:
            merged = long_df
        else:
            merged = merged.merge(long_df, on=['date', 'stock'], how='outer')

    if merged is None or merged.empty:
        print("⚠️  合并行情数据为空")
        return None

    merged = merged.sort_values(['stock', 'date']).reset_index(drop=True)

    # 计算市值
    if 'close' in merged.columns and 'outstanding_share' in merged.columns:
        merged['mkt_cap'] = merged['close'] * merged['outstanding_share']
    else:
        merged['mkt_cap'] = pd.NA

    # 获取行业和概念数据
    try:
        code_list = merged['stock'].dropna().astype(str).unique().tolist()
        ind_map = data.get_industry_category(code_list) if code_list else {}
        cpt_map = data.get_concept_categories(code_list) if code_list else {}
    except Exception as e:
        print(f"⚠️  获取行业概念数据失败: {e}")
        ind_map, cpt_map = {}, {}

    def _code_industry(c: str) -> str:
        if isinstance(ind_map, dict):
            return ind_map.get(c) or industry_default
        return industry_default

    def _code_concepts(c: str) -> str:
        vals = []
        if isinstance(cpt_map, dict):
            vals = cpt_map.get(c) or []
        return ','.join([str(v) for v in vals if v]) if vals else ''

    merged['industry'] = merged['stock'].astype(str).map(_code_industry)
    merged['concepts'] = merged['stock'].astype(str).map(_code_concepts)

    # --------- 2. 加载OSS因子数据 ---------
    df_factors = data.read_factor_data(
        codes=codes,
        start_date=start_date,
        end_date=end_date,
        factors=factors if (factors and len(factors)>0) else None,
        base_path="uploads"
    )

    if df_factors is not None and not df_factors.empty:
        df_factors = df_factors.reset_index()
        df_factors.rename(columns={"code": "stock"}, inplace=True)

        # 统一股票代码格式
        def _to_six_digit(s: pd.Series) -> pd.Series:
            s = s.astype(str).str.upper()
            s = (s.str.replace('.XSHG', '', regex=False)
                 .str.replace('.XSHE', '', regex=False)
                 .str.replace('.XBJ', '', regex=False))
            extracted = s.str.extract(r'(\d{6})')[0]
            s = extracted.where(extracted.notna(), s)
            return s

        merged['stock'] = _to_six_digit(merged['stock'])
        df_factors['stock'] = _to_six_digit(df_factors['stock'])

        merged = merged.merge(df_factors, on=["date", "stock"], how="outer")

    # --------- 3. 处理TALIB因子数据 ---------
    if include_talib:
        # 默认TALIB因子列表
        default_talib_factors = [
            'TALIB_HT_DCPERIOD',
            'TALIB_MACD_12_26_9',
            'TALIB_MACDEXT_12_26_9_0_0_0',
            'TALIB_MACDFIX_9'
        ]
        
        if compute_talib:
            print("正在直接计算TALIB因子...")
            # 导入因子计算器
            import factor_old.factor_calculator as fc
            
            talib_list = talib_factors if talib_factors else default_talib_factors
            
            for talib_factor in talib_list:
                try:
                    print(f"  计算 {talib_factor}...")
                    calc = fc.create_factor_calculator(factor_name=talib_factor)
                    
                    # 为每只股票计算因子
                    all_factor_data = []
                    unique_stocks = merged['stock'].dropna().unique()
                    
                    for stock in unique_stocks:
                        try:
                            factor_series = calc.calculate(stock, start_date, end_date)
                            if not factor_series.empty:
                                factor_series = factor_series.dropna()
                                if not factor_series.empty:
                                    stock_df = pd.DataFrame({
                                        'date': factor_series.index,
                                        'stock': stock,
                                        talib_factor: factor_series.values
                                    })
                                    all_factor_data.append(stock_df)
                        except Exception as e:
                            print(f"    ⚠️  {stock} 计算失败: {e}")
                            continue
                    
                    if all_factor_data:
                        factor_df = pd.concat(all_factor_data, ignore_index=True)
                        merged = merged.merge(factor_df, on=['date', 'stock'], how='left')
                        
                        non_null = merged[talib_factor].notna().sum()
                        print(f"  ✓ {talib_factor}: 计算了 {non_null} 条记录")
                    else:
                        print(f"  ⚠️ {talib_factor}: 无数据")
                        
                except Exception as e:
                    print(f"  ❌ {talib_factor}: 计算失败 {e}")
        else:
            print("正在加载TALIB因子数据...")
            
            talib_list = talib_factors if talib_factors else default_talib_factors
            
            for talib_factor in talib_list:
                try:
                    # 尝试从已导出的TALIB因子文件中加载
                    talib_file = f'exported_data/{talib_factor}_{start_date.replace("-", "_")}_{end_date.replace("-", "_")}.csv'
                    if not os.path.exists(talib_file):
                        # 尝试另一种命名格式（使用连字符）
                        talib_file = f'exported_data/{talib_factor}_{start_date}_{end_date}.csv'
                    if os.path.exists(talib_file):
                        talib_df = pd.read_csv(talib_file)
                        talib_df['date'] = pd.to_datetime(talib_df['date'])
                        talib_df['code'] = talib_df['code'].astype(str).str.zfill(6)

                        merged = merged.merge(
                            talib_df[['date', 'code', 'factor_value']],
                            left_on=['date', 'stock'],
                            right_on=['date', 'code'],
                            how='left'
                        )

                        merged = merged.rename(columns={'factor_value': talib_factor})
                        merged = merged.drop('code', axis=1, errors='ignore')

                        non_null = merged[talib_factor].notna().sum()
                        print(f"  ✓ {talib_factor}: 加载了 {non_null} 条记录")

                    else:
                        print(f"  ⚠️ {talib_factor}: 文件不存在 {talib_file}")

                except Exception as e:
                    print(f"  ❌ {talib_factor}: 加载失败 {e}")

    # --------- 4. 整理最终数据 ---------
    base_cols = ['date', 'stock', 'open', 'high', 'low', 'close', 'volume', 'amount', 'mkt_cap', 'industry', 'concepts']

    # 获取所有因子列
    factor_cols = []
    if df_factors is not None and not df_factors.empty:
        factor_cols.extend([c for c in df_factors.columns if c not in base_cols and c not in ('date', 'stock')])

    if include_talib:
        talib_cols = talib_factors if talib_factors else default_talib_factors
        factor_cols.extend([c for c in talib_cols if c in merged.columns])

    # 保持指定因子的顺序
    if factors and len(factors) > 0:
        factor_cols = [c for c in factors if c in factor_cols] + [c for c in factor_cols if c not in factors]

    final_cols = base_cols + factor_cols

    # 补全缺失的列
    for c in final_cols:
        if c not in merged.columns:
            merged[c] = pd.NA

    out_df = merged[final_cols].sort_values(['date', 'stock']).reset_index(drop=True)

    # 数据质量检查
    price_complete = out_df[['open', 'high', 'low', 'close']].notna().all(axis=1).sum()
    price_pct = price_complete / len(out_df) * 100 if len(out_df) > 0 else 0

    # 保存文件
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'formatted_data.csv')
    out_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n✓ 完整格式化数据已导出: {output_file}")
    print(f"  记录数: {len(out_df)}")
    print(f"  股票数: {out_df['stock'].nunique()}")
    print(f"  日期范围: {out_df['date'].min()} ~ {out_df['date'].max()}")
    print(f"  价格完整性: {price_complete}/{len(out_df)} ({price_pct:.1f}%)")
    print(f"  OSS因子数: {len([c for c in factor_cols if not c.startswith('TALIB_')])}")
    print(f"  TALIB因子数: {len([c for c in factor_cols if c.startswith('TALIB_')])}")

    return output_file

def main():
    parser = argparse.ArgumentParser(
        description='导出价格数据和因子数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 导出最近3个月的格式化数据（默认模式）
  python export_data.py --stocks 000001 000002 600000 --factors VOL10 VSTD10
  
  # 导出指定日期范围的格式化数据
  python export_data.py --stocks 000001 000002 --factors VOL10 \\
      --start 2024-01-01 --end 2024-03-31 --mode formatted
  
  # 导出包含TALIB因子的完整数据（直接计算）
  python export_data.py --stocks 000001 000002 --factors VOL10 \\
      --include-talib --talib-factors TALIB_MACD_12_26_9 TALIB_HT_DCPERIOD --compute-talib
  
  # 仅导出基础行情数据（不含因子）
  python export_data.py --stocks 000001 --mode formatted --factors ""
  
  # 指定输出目录
  python export_data.py --stocks 000001 --factors VOL10 --output ./my_data
        """
    )
    
    parser.add_argument(
        '--stocks',
        nargs='+',
        required=True,
        help='股票代码列表（别墅后缀，如: 000001 000002 600000）'
    )
    
    parser.add_argument(
        '--factors',
        nargs='+',
        required=True,
        help='因子名称列表（如: VOL10 VSTD10）'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        default=None,
        help='开始日期 (YYYY-MM-DD)，默认：最近3个月的开始日期'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        default=None,
        help='结束日期 (YYYY-MM-DD)，默认：今天'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./exported_data',
        help='输出目录（默认: ./exported_data）'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['separate', 'combined', 'both', 'formatted', 'formatted_talib'],
        default='formatted',
        help='导出模式: separate=分别导出, combined=合并导出, both=都导出, formatted=格式化导出, formatted_talib=格式化导出+TALIB因子'
    )
    
    parser.add_argument(
        '--include-talib',
        action='store_true',
        help='是否包含TALIB技术指标因子（仅在formatted模式下有效）'
    )
    
    parser.add_argument(
        '--talib-factors',
        nargs='+',
        default=None,
        help='TALIB因子名称列表（如: TALIB_MACD_12_26_9 TALIB_HT_DCPERIOD），None时使用默认列表'
    )
    
    parser.add_argument(
        '--compute-talib',
        action='store_true',
        help='直接计算TALIB因子（不从文件加载）'
    )
    
    args = parser.parse_args()
    
    # 确定日期范围
    if args.start and args.end:
        start_date = args.start
        end_date = args.end
    else:
        start_date, end_date = get_last_3_months()
        print(f"使用默认日期范围: {start_date} ~ {end_date}")
    
    # 标准化股票代码（确保是列表）
    codes = args.stocks
    if isinstance(codes, str):
        codes = [codes]
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"数据导出配置")
    print(f"{'='*60}")
    print(f"股票代码: {codes}")
    print(f"因子列表: {args.factors}")
    print(f"日期范围: {start_date} ~ {end_date}")
    print(f"输出目录: {args.output}")
    print(f"导出模式: {args.mode}")
    print(f"{'='*60}\n")
    
    # 导出数据
    if args.mode in ['separate', 'both']:
        price_file = export_price_data(codes, start_date, end_date, args.output)
        factor_file = export_factor_data(codes, args.factors, start_date, end_date, args.output)
    
    if args.mode in ['combined', 'both']:
        combined_file = export_combined_data(codes, args.factors, start_date, end_date, args.output)

    if args.mode in ['formatted', 'formatted_talib', 'custom']:
        # 向后兼容旧的custom模式
        include_talib = args.mode == 'formatted_talib' or args.include_talib
        
        export_formatted_csv(
            codes=codes,
            start_date=start_date,
            end_date=end_date,
            output_dir=args.output,
            factors=args.factors,
            industry_default='Unknown',
            include_talib=include_talib,
            talib_factors=args.talib_factors,
            compute_talib=args.compute_talib
        )
    
    print(f"\n{'='*60}")
    print(f"导出完成！")
    print(f"输出目录: {args.output}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()


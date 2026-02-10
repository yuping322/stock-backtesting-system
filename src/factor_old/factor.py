# ===========================================================
#  单因子 / 多因子 Alphalens 一键检验 + 自动打分脚本
#  适用平台：JoinQuant / RiceQuant 等支持 JQData 的环境
# ===========================================================
# 注意：不在这里设置 Agg backend，因为 alphalens 需要交互式 backend 才能正确生成图表
# 如果需要在无 GUI 环境运行，可以在调用方根据实际情况设置 backend
import matplotlib
# matplotlib.use('Agg')  # 注释掉，让 matplotlib 自动选择 backend（通常是交互式的）

import alphalens as al
import pandas as pd
import numpy as np
import warnings
import argparse
import json
import sys
import os
import time
from typing import List, Optional
from pathlib import Path

# 添加父目录到路径，以便导入项目模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 添加 factor 目录到路径
import data
try:
    from factor_old.factor_calculator import FileFactorCalculator
except ImportError:
    # 如果作为模块导入失败，尝试直接导入
    from factor_calculator import FileFactorCalculator

# 导入因子计算器
from factor_calculator import (
    FactorCalculator,
    OHLCVFactorCalculator,
    CustomFactorCalculator,
    BuiltinFactorCalculator,
    create_factor_calculator
)

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='单因子/多因子 Alphalens 一键检验 + 自动打分',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置
  python factor.py
  
  # 指定回测区间和股票池
  python factor.py --start 2024-01-01 --end 2024-12-31 --stock-pool 000510.XSHG
  
  # 指定因子和调仓周期
  python factor.py --factors VOL10 VSTD10 --periods 5 10 15
  
  # 指定分位数和滚动窗口
  python factor.py --quantiles 5 --roll-win 30
        """
    )
    
    # 回测区间
    parser.add_argument('--start', type=str, default='2024-09-25',
                       help='回测开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-10-14',
                       help='回测结束日期 (YYYY-MM-DD)')
    
    # 股票池
    parser.add_argument('--stock-pool', type=str, default='000510.XSHG',
                       help='股票池：指数代码 或 "stock"（全市场）')
    
    # 因子
    parser.add_argument('--factors', nargs='+', default=['VOL10', 'single_day_VPT_12'],
                       help='要检验的因子列表')
    parser.add_argument('--factor-dir', type=str, default=None,
                       help='因子文件目录，会自动查找包含指定因子的CSV文件')
    
    # Alphalens 参数
    parser.add_argument('--quantiles', type=int, default=10,
                       help='分组数量')
    parser.add_argument('--periods', nargs='+', type=int, default=[5, 10, 15],
                       help='调仓周期（天）')
    
    # 数据清洗开关
    parser.add_argument('--fillna', type=int, default=0,
                       help='是否填充缺失值 (0=否, 1=是)')
    parser.add_argument('--winsorize', type=int, default=0,
                       help='是否异常值处理 (0=否, 1=是)')
    parser.add_argument('--neutralize', type=int, default=0,
                       help='是否中性化 (0=否, 1=是)')
    parser.add_argument('--standardize', type=int, default=0,
                       help='是否标准化 (0=否, 1=是)')
    
    # 滚动窗口和监控
    parser.add_argument('--roll-win', type=int, default=60,
                       help='滚动窗口交易日数')
    parser.add_argument('--monitor-csv', type=str, default='monitor.csv',
                       help='监控结果CSV文件路径')
    parser.add_argument('--last-only', action='store_true',
                       help='只输出最新一期 (默认: False)')
    
    return parser.parse_args()


class CFG:
    """配置类，从命令行参数初始化"""
    def __init__(self, args):
        self.START = args.start
        self.END = args.end
        self.STOCK_POOL = args.stock_pool
        self.FACTORS = args.factors
        self.QUANTILES = args.quantiles
        self.PERIODS = args.periods
        self.FREQ = pd.tseries.offsets.Day()  # 使用pandas offset对象而不是字符串
        self.CLEAN = {
            'fillna': args.fillna,
            'winsorize': args.winsorize,
            'neutralize': args.neutralize,
            'standardize': args.standardize
        }
        self.ROLL_WIN = args.roll_win
        self.MONITOR_CSV = args.monitor_csv
        self.LAST_ONLY = args.last_only
        # 因子文件目录（可选）
        self.FACTOR_DIR = getattr(args, 'factor_dir', None)
        # 股票数量限制（用于测试，None表示使用全部股票）
        self.MAX_STOCKS = getattr(args, 'max_stocks', None)
        # 因子元数据（暂时保留默认值）
        self.FACTOR_META = {'VSTD10': {'direct': 1, 'weight': 1}}


    
CSRC_NAME = {
    'A01': '农业',
    'A02': '林业',
    'A03': '畜牧业',
    'A04': '渔业',
    'A05': '农林牧渔服务业',
    'B06': '煤炭开采和洗选业',
    'B07': '石油和天然气开采业',
    'B08': '黑色金属矿采选业',
    'B09': '有色金属矿采选业',
    'B10': '非金属矿采选业',
    'B11': '开采辅助活动',
    'B12': '其他采矿业',
    'C13': '农副食品加工业',
    'C14': '食品制造业',
    'C15': '酒饮料和精制茶制造业',
    # ……按需补全
}
    
def rolling_monitor(factor_name, ic_series, tb_ret_series, period_days, cfg):
    """每天追加一行滚动指标"""
    import os
    
    # 自适应滚动窗口：使用实际数据长度，但不超过配置的窗口大小
    data_len = len(ic_series)
    roll_win = min(cfg.ROLL_WIN, data_len)
    
    # 如果数据为空或太短，返回默认值
    if data_len == 0:
        roll_ic = np.nan
        roll_ir = np.nan
        roll_t = np.nan
        top_std = np.nan
        neg_day = np.nan
    else:
        # 使用实际窗口大小，最小值为1
        roll_win = max(1, roll_win)
        roll_ic = ic_series.rolling(roll_win, min_periods=1).mean().iloc[-1]
        roll_std = ic_series.rolling(roll_win, min_periods=1).std().iloc[-1]
        roll_ir = roll_ic / roll_std if roll_std != 0 else np.nan
        roll_t = roll_ic / roll_std * np.sqrt(roll_win) if roll_std != 0 else np.nan
        top_std = tb_ret_series.rolling(roll_win, min_periods=1).std().iloc[-1]
        neg_day = (tb_ret_series < 0).rolling(roll_win, min_periods=1).mean().iloc[-1]

    df = pd.DataFrame({
        'date'      : [pd.Timestamp.now().normalize()],
        'factor'    : [factor_name],
        'period'    : [period_days],
        'roll_ic'   : [roll_ic],
        'roll_ir'   : [roll_ir],
        'roll_t'    : [roll_t],
        'top_std'   : [top_std],
        'neg_day'   : [neg_day]
    })
    # 追加写
    hdr = not os.path.exists(cfg.MONITOR_CSV)
    df.to_csv(cfg.MONITOR_CSV, mode='a', header=hdr, index=False)
    return df.iloc[0]   # 返回最新一行，给主函数打印


# -----------------------------------------------------------
# 2. 打分函数（动态阈值）
# -----------------------------------------------------------
def quick_score(factor_name, factor_data, period_days, cfg):
    """
    根据调仓周期 period_days 动态调整阈值
    """
    from alphalens.performance import (
        factor_information_coefficient,
        factor_returns,
        quantile_turnover,
        mean_return_by_quantile
    )
    import pandas as pd

    # 1. IC
    try:
        ic = factor_information_coefficient(factor_data)
        if ic.empty or len(ic) == 0:
            print(f'⚠️  {factor_name} IC 数据为空')
            return None, None, None, None, None
        
        # 确保 ic 是数值类型
        ic = ic.apply(pd.to_numeric, errors='coerce')
        
        ic_mean = ic.mean().iloc[0]
        ic_std = ic.std().iloc[0]
        icir = ic_mean / ic_std if ic_std != 0 and not np.isnan(ic_std) else np.nan
    except Exception as e:
        print(f'⚠️  {factor_name} IC 计算失败: {e}')
        return None, None, None, None, None

    # -------- 2. Top-Bottom 收益 --------
    ret = factor_returns(factor_data)
    ret_col = ret.columns[cfg.PERIODS.index(period_days)]
    tb_ret = ret[ret_col]
    ann_ret = tb_ret.mean() * 252
    max_dd = (tb_ret.cumsum().cummax() - tb_ret.cumsum()).max()

    # 3. Top quintile 换手率（跨版本通用自算版）
    # 兼容所有版本：Top 分组 = quantile=cfg.QUANTILES
    quant_col = factor_data['factor_quantile']
    # 取出 Top 组（最高分位）每天的资产列表
    top_quantile = cfg.QUANTILES
    top = (quant_col[quant_col == top_quantile]
           .reset_index()
           .groupby('date')['asset']
           .apply(list))

    # 计算换手率：1 - 交集/前一期
    # 先把 top 和 top.shift() 转换成集合
    top_sets = top.apply(lambda x: set(x) if isinstance(x, list) else set())
    shift_sets = top.shift().apply(lambda x: set(x) if isinstance(x, list) else set())

    # 逐元素交集
    intersection = pd.Series(
        [len(a & b) for a, b in zip(top_sets, shift_sets)],
        index=top.index
    )

    # 分母（上一期的 Top 数量）
    denom = shift_sets.apply(len)

    # 换手率
    turnover_rate = 1 - (intersection / denom)
    top_turnover = turnover_rate.dropna().mean()

    
    
    # -------- 4. 单调性 --------
    mean_ret, _ = mean_return_by_quantile(
        factor_data, by_group=False, demeaned=True)
    mean_col = mean_ret.columns[cfg.PERIODS.index(period_days)]
    
    # 确保索引长度匹配
    if len(mean_ret.index) < 2:
        mono = np.nan
    else:
        # 使用实际的分位数索引长度
        actual_quantiles = len(mean_ret.index)
        mono = mean_ret[mean_col].corr(pd.Series(range(1, actual_quantiles + 1),
                                                     index=mean_ret.index))

    # 5. 动态阈值表（优化后的阈值）
    thresholds = {
        1:  {'ic': 0.03, 'ir': 0.8, 'ret': 0.15, 'mono': 0.85, 'turn': 0.60,
              'tau': 8, 'sharpe': 1.0, 'net_ir': 0.2, 'cap': 3},
        5:  {'ic': 0.05, 'ir': 1.0, 'ret': 0.12, 'mono': 0.85, 'turn': 0.40,
              'tau': 10, 'sharpe': 0.8, 'net_ir': 0.3, 'cap': 5},
        10: {'ic': 0.05, 'ir': 1.0, 'ret': 0.10, 'mono': 0.80, 'turn': 0.30,
              'tau': 12, 'sharpe': 0.8, 'net_ir': 0.3, 'cap': 5},
        15: {'ic': 0.05, 'ir': 1.0, 'ret': 0.08, 'mono': 0.75, 'turn': 0.25,
              'tau': 15, 'sharpe': 0.7, 'net_ir': 0.3, 'cap': 5}
    }
    th = thresholds.get(period_days, thresholds[5])

    
    # -------- 6. 新增指标 --------
    # 6-1 IC 半衰期
    def ic_half_life(ic_series):
        """计算 IC 半衰期"""
        from scipy.optimize import curve_fit
        ic_series = ic_series.dropna()
        if len(ic_series) < 20:
            return np.nan
        
        # 使用绝对值进行拟合，避免负值问题
        ic_abs = ic_series.abs()
        
        # 如果平均值为0或太小，跳过拟合
        if ic_abs.mean() < 0.001:
            return np.nan
        
        t = np.arange(len(ic_abs))
        def f(t, ic0, tau): 
            return ic0 * np.exp(-t / tau)
        
        try:
            # 使用合理的初始值
            initial_tau = 20
            initial_ic0 = ic_abs.iloc[0] if ic_abs.iloc[0] > 0 else ic_abs.mean()
            p, _ = curve_fit(f, t, ic_abs.values, p0=(initial_ic0, initial_tau), maxfev=1000)
            tau = p[1]
            
            # 检查结果是否合理（应该在0到1000之间）
            if 0 < tau < 1000:
                return tau
            else:
                return np.nan
        except:
            return np.nan

    tau = ic_half_life(ic.iloc[:, 0])

    # 6-2 Q5-Q1 夏普（直接用 ret 已算好）
    q5q1_sharpe = tb_ret.mean() / tb_ret.std() * np.sqrt(252)

    # 6-3 扣费后净 IR（双边 0.3%）
    cost_rate = 0.003 * 2
    # 假设每期调仓一次，换手 = top_turnover
    std_ret = tb_ret.std()
    if std_ret == 0 or np.isnan(std_ret):
        net_ir = np.nan
    else:
        net_ir = (tb_ret.mean() - top_turnover * 0.003 * 2) / std_ret * np.sqrt(252)

    # 6-4 容量指标：成交额加权换手率
    # 需要 daily 成交额；alphalens 的 clean 里没带，这里用 mock，真实环境替换成 get_money_flow
    def mock_turnover_cap(factor_data):
        # 用 clean 里的因子分位 dataframe 反推 top 组合每日股票列表
        quant_df = factor_data['factor_quantile']
        top_assets = (quant_df[quant_df == cfg.QUANTILES]
                      .reset_index()
                      .groupby('date')['asset'].apply(list))
        # mock 成交额：假设每只 1e8，真实场景用 get_money_flow
        mock_amt = 1e8
        amt_weighted = top_assets.apply(lambda x: mock_amt * len(x))
        # 容量 = 日均可成交金额 / 换手率
        avg_amt = amt_weighted.mean()
        return avg_amt / (top_turnover + 1e-6)
    capacity = mock_turnover_cap(factor_data) / 1e8

    
    def grade(val, good, direction=1):
        return (val * direction) >= good

    
    scores = {
        'IC均值':     (ic_mean, grade(abs(ic_mean), th['ic'])),
        'IR比率':     (icir,    grade(icir, th['ir'])),
        '多空年化':   (ann_ret, grade(ann_ret, th['ret'])),
        '单调性':     (mono,    grade(mono, th['mono'])),
        'Top换手率':  (top_turnover, grade(top_turnover, th['turn'], 1)) ,
        'IC半衰期(τ)': (tau, grade(tau, th['tau'])),               # 使用动态阈值
        'Q5-Q1夏普': (q5q1_sharpe, grade(q5q1_sharpe, th['sharpe'])),
        '扣费净IR': (net_ir, grade(net_ir, th['net_ir'])),
        '容量(亿元)': (capacity, grade(capacity, th['cap']))        # 使用动态阈值
    }

    passed = sum(scores[k][1] for k in scores)
    level = '优秀' if passed >= 4 else ('良好' if passed >= 3 else '一般')

    # quick_score 函数末尾加一行
    return scores, level, ic.iloc[:, 0], tb_ret, top_turnover


# -----------------------------------------------------------
# 3. 结果类
# -----------------------------------------------------------
class FactorTestResult:
    """因子检验结果"""
    
    def __init__(self):
        self.factor_name = None
        self.period = None
        self.scores = {}
        self.level = None
        self.ic_series = None
        self.ret_series = None
        self.top_turnover = None
        self.rolling_monitor = None
        self.status_flag = None
        self.clean_data = None
        
    def to_dict(self):
        """转换为字典"""
        return {
            'factor_name': self.factor_name,
            'period': self.period,
            'scores': self.scores,
            'level': self.level,
            'status_flag': self.status_flag,
            'top_turnover': self.top_turnover,
        }


# -----------------------------------------------------------
# 4. 主类
# -----------------------------------------------------------
class FactorTester:
    def __init__(self, cfg, custom_factors=None):
        """
        初始化因子检验器
        
        Args:
            cfg: 配置对象
            custom_factors: 自定义因子字典，格式为 {factor_name: FactorCalculator}
        """
        self.cfg = cfg
        self.stocks = []
        self.custom_factors = custom_factors or {}

    def _get_stocks_for_calculation(self):
        """获取用于计算的股票列表，根据配置决定是否限制数量"""
        if self.cfg.MAX_STOCKS is not None:
            stocks = self.stocks[:self.cfg.MAX_STOCKS]
            print(f'[配置] 限制使用前 {self.cfg.MAX_STOCKS} 只股票进行计算')
        else:
            stocks = self.stocks
            print(f'[配置] 使用全部 {len(stocks)} 只股票进行计算')
        return stocks

    # ---------- 取股票 ----------
    def get_stocks(self):
        import time
        start_time = time.time()
        t0 = time.strftime('%H:%M:%S')
        print(f'[get_stocks开始] {t0}')
        
        if self.cfg.STOCK_POOL == 'stock':
            # 获取A股所有股票代码
            try:
                self.stocks = data.get_index_stocks("small")
                print(f'[get_stocks] 获取到A股股票 {len(self.stocks)} 只')
            except Exception as e:
                print(f'[get_stocks] 获取A股股票失败: {e}，使用空列表')
                self.stocks = []
        else:
            # 获取指数成分股
            symbol = self.cfg.STOCK_POOL.split('.')[0]
            self.stocks = data.get_index_stocks(symbol)
        print(f'[{t0}] 股票池({len(self.stocks)}) 加载完成')
        elapsed = time.time() - start_time
        print(f'[get_stocks完成] 耗时: {elapsed:.2f}秒')

    # ---------- 取因子 ----------
    def get_factors(self):
        print('正在计算因子...')
        
        # 如果使用文件因子，检查并更新日期范围和股票列表
        if self.custom_factors:
            file_date_ranges = []
            file_stocks_list = []
            for calc in self.custom_factors.values():
                if isinstance(calc, FileFactorCalculator):
                    file_start, file_end = calc.get_file_date_range()
                    if file_start and file_end:
                        file_date_ranges.append((file_start, file_end))
                    # 从文件因子中获取股票列表
                    stocks = calc.get_file_stocks()
                    if stocks:
                        file_stocks_list.extend(stocks)
            
            # 更新日期范围
            if file_date_ranges:
                # 使用所有文件因子中最小的日期范围（取交集）
                min_start = max(r[0] for r in file_date_ranges)
                max_end = min(r[1] for r in file_date_ranges)
                if min_start <= max_end:
                    # 更新配置中的日期范围
                    self.cfg.START = min_start.strftime('%Y-%m-%d')
                    self.cfg.END = max_end.strftime('%Y-%m-%d')
                    print(f'[文件因子] 使用因子文件日期范围: {self.cfg.START} 到 {self.cfg.END}')
            
            # 如果股票列表为空，从文件因子中获取股票列表
            if not self.stocks and file_stocks_list:
                # 去重并排序
                self.stocks = sorted(list(set(file_stocks_list)))
                print(f'[文件因子] 从因子文件中获取到 {len(self.stocks)} 只股票')
        
        factors = {}
        for f in self.cfg.FACTORS:
            try:
                # 优先使用自定义因子计算器
                if f in self.custom_factors:
                    factor_data = {}
                    calculator = self.custom_factors[f]
                    
                    # 使用配置中的日期范围（已在run方法中根据文件因子更新）
                    for stock in self._get_stocks_for_calculation():
                        try:
                            factor_series = calculator.calculate(stock, self.cfg.START, self.cfg.END)
                            if factor_series is not None and len(factor_series) > 0:
                                for date, val in factor_series.items():
                                    factor_data[(pd.Timestamp(date), stock)] = val
                        except Exception as e:
                            print(f'  计算因子 {f} 股票 {stock} 失败: {e}')
                    if factor_data:
                        # 直接从字典创建 Series，会自动创建 MultiIndex
                        factors[f] = pd.Series(factor_data)
                        # 设置索引名称
                        factors[f].index.names = ['date', 'asset']
                    else:
                        print(f'⚠️  因子 {f} 无数据')
                
                # 否则使用内置因子（兼容旧代码）
                elif f == 'VOL10':
                    factor_data = {}
                    for stock in self._get_stocks_for_calculation():
                        df = data.load_oss_complex_stocks([stock], start=self.cfg.START, end=self.cfg.END, fields='all')
                        if 'volume' in df:
                            vol = df['volume'][stock]
                            vol_ma10 = vol.rolling(10).mean()
                            for date, val in vol_ma10.items():
                                factor_data[(pd.Timestamp(date), stock)] = val
                    if factor_data:
                        factors[f] = pd.Series(factor_data)
                        factors[f].index.names = ['date', 'asset']
                elif f == 'single_day_VPT_12':
                    factor_data = {}
                    for stock in self._get_stocks_for_calculation():
                        df = data.load_oss_complex_stocks([stock], start=self.cfg.START, end=self.cfg.END, fields='all')
                        if 'close' in df and 'volume' in df:
                            close = df['close'][stock]
                            vol = df['volume'][stock]
                            ret = close.pct_change()
                            vpt = (ret * vol).rolling(12).sum()
                            for date, val in vpt.items():
                                factor_data[(pd.Timestamp(date), stock)] = val
                    if factor_data:
                        factors[f] = pd.Series(factor_data)
                        factors[f].index.names = ['date', 'asset']
                else:
                    # 尝试从 OSS 读取因子
                    print(f'尝试从 OSS 读取因子: {f}')
                    try:
                        factor_series = data.factor_for_al(
                            codes=self._get_stocks_for_calculation(),
                            start_date=self.cfg.START,
                            end_date=self.cfg.END,
                            factor_name=f
                        )
                        if factor_series is not None and len(factor_series) > 0:
                            factors[f] = factor_series
                            print(f'✓ 成功从 OSS 读取因子 {f}，共 {len(factor_series)} 条数据')
                        else:
                            print(f'⚠️  因子 {f} 在 OSS 中无数据')
                    except Exception as e:
                        print(f'❌ 从 OSS 读取因子 {f} 失败：{e}')
            except Exception as e:
                print(f'❌ 因子 {f} 计算失败：{e}')
        return factors

    # ---------- 取价格 ----------
    def get_price(self):
        import time
        start_time = time.time()
        print(f'[get_price开始] 正在拉取行情... - {time.strftime("%H:%M:%S")}')
        
        # 使用 data.py 获取价格数据
        try:
            price = data.load_oss_stocks(
                codes=self._get_stocks_for_calculation(),  # 限制前10只股票测试
                start=self.cfg.START,
                end=self.cfg.END
            )
            price.index.name = 'date'
            elapsed = time.time() - start_time
            print(f'[get_price完成] 耗时: {elapsed:.2f}秒')
            return price
        except Exception as e:
            elapsed = time.time() - start_time
            print(f'[get_price失败] 耗时: {elapsed:.2f}秒 - 错误: {e}')
            return pd.DataFrame()

    # ---------- 取行业/概念分组 ----------
    def get_industry_series(self):
        import time
        start_time = time.time()
        print(f'[get_industry开始] 正在拉取行业... - {time.strftime("%H:%M:%S")}')

        # 获取交易日历（按日展开标签）
        dates = pd.date_range(self.cfg.START, self.cfg.END, freq='D')

        stocks = self._get_stocks_for_calculation()

        # 优先用概念主标签，其次行业，没有则 Other
        try:
            concept_map = data.get_concept_categories(stocks)
        except Exception:
            concept_map = {s: [] for s in stocks}
        try:
            industry_map = data.get_industry_category(stocks)
        except Exception:
            industry_map = {s: None for s in stocks}

        def pick_group(code: str) -> str:
            concepts = []
            if isinstance(concept_map, dict):
                concepts = concept_map.get(code, []) or []
            # 取第一个概念作为主标签
            if concepts:
                return str(concepts[0])
            # 回退行业
            if isinstance(industry_map, dict):
                ind = industry_map.get(code)
                if ind:
                    return str(ind)
            return 'Other'

        group_dict = {}
        for code in stocks:
            grp = pick_group(code)
            for d in dates:
                group_dict[(pd.Timestamp(d), code)] = grp

        elapsed = time.time() - start_time
        print(f'[get_industry完成] 耗时: {elapsed:.2f}秒')
        return pd.Series(group_dict, name='group')

    # ---------- 主流程 ----------
    def run(self, plot=False):
        """
        运行因子检验（处理所有因子）
        
        Args:
            plot: 是否画图，默认 False
            
        Returns:
            list: FactorTestResult 列表
        """
        # 如果使用文件因子，跳过get_stocks()，直接依赖get_factors()从文件因子中获取股票列表
        # 否则，先获取股票列表
        if not self.custom_factors:
            # 没有文件因子，先获取股票列表
            if not self.stocks:
                self.get_stocks()
                # 如果股票列表仍然为空，给出警告
                if not self.stocks:
                    print(f'⚠️  警告：股票列表为空，无法进行因子验证')
                    print(f'   请检查：')
                    print(f'   1. OSS连接是否正常（用于获取股票池）')
                    print(f'   2. --stock-pool 参数是否正确')
        
        # 获取因子（get_factors 内部会处理文件因子的股票列表和日期范围）
        factors = self.get_factors()
        price = self.get_price()
        industry = self.get_industry_series()

        results = {}
        for f, factor_series in factors.items():
            clean = al.utils.get_clean_factor_and_forward_returns(
                factor_series,
                prices=price,
                groupby=industry,
                quantiles=self.cfg.QUANTILES,
                periods=self.cfg.PERIODS,
                max_loss=3
            )
            results[f] = clean

        # 存储所有检验结果
        all_results = []

        # 打分 & tear-sheet
        for f, clean in results.items():
            all_results.extend(self._process_single_factor(f, clean, plot))

        print('全部完成！')
        return all_results
    
    def get_single_factor(self, factor_name):
        """
        获取单个因子数据
        
        Args:
            factor_name: 因子名称
            
        Returns:
            pd.Series: 因子序列，index 为 (date, asset)
        """
        import time
        start_time = time.time()
        print(f'[get_single_factor开始] 正在计算因子: {factor_name}... - {time.strftime("%H:%M:%S")}')
        
        try:
            # 优先使用自定义因子计算器
            if factor_name in self.custom_factors:
                factor_data = {}
                calculator = self.custom_factors[factor_name]
                print(f'[自定义因子] 开始计算 {factor_name} 因子，股票数量: {len(self._get_stocks_for_calculation())}')
                
                for i, stock in enumerate(self._get_stocks_for_calculation()):
                    stock_start = time.time()
                    print(f'[自定义因子] 计算股票 {stock} ({i+1}/{len(self._get_stocks_for_calculation())}) - {time.strftime("%H:%M:%S")}')
                    
                    try:
                        factor_series = calculator.calculate(stock, self.cfg.START, self.cfg.END)
                        
                        stock_elapsed = time.time() - stock_start
                        print(f'[自定义因子] 股票 {stock} 计算完成 - 耗时: {stock_elapsed:.2f}秒')
                        
                        if factor_series is not None and len(factor_series) > 0:
                            for date, val in factor_series.items():
                                factor_data[(pd.Timestamp(date), stock)] = val
                            print(f'[自定义因子] 股票 {stock} 有效数据点: {len(factor_series)}')
                        else:
                            print(f'[自定义因子] 股票 {stock} 无有效数据')
                    except Exception as e:
                        stock_elapsed = time.time() - stock_start
                        print(f'[自定义因子] 股票 {stock} 计算失败 - 耗时: {stock_elapsed:.2f}秒 - 错误: {e}')
                
                print(f'[自定义因子] 所有股票处理完成，总因子数据点: {len(factor_data)}')
                if factor_data:
                    factor_series = pd.Series(factor_data)
                    factor_series.index.names = ['date', 'asset']
                    elapsed = time.time() - start_time
                    print(f'[get_single_factor完成] 自定义因子 {factor_name} - 耗时: {elapsed:.2f}秒')
                    return factor_series
            
            # 尝试内置因子
            elif factor_name == 'VOL10':
                factor_data = {}
                print(f'[VOL10] 开始计算VOL10因子，股票数量: {len(self._get_stocks_for_calculation())}')
                for i, stock in enumerate(self._get_stocks_for_calculation()):
                    stock_start = time.time()
                    print(f'[VOL10] 拉取股票 {stock} 数据 ({i+1}/{len(self._get_stocks_for_calculation())}) - {time.strftime("%H:%M:%S")}')
                    
                    df = data.load_oss_complex_stocks([stock], start=self.cfg.START, end=self.cfg.END, fields='all')
                    
                    stock_elapsed = time.time() - stock_start
                    print(f'[VOL10] 股票 {stock} 数据拉取完成 - 耗时: {stock_elapsed:.2f}秒')
                    
                    if isinstance(df, dict) and 'volume' in df and not df['volume'].empty:
                        vol = df['volume'][stock]
                        vol_ma10 = vol.rolling(10).mean()
                        for date, val in vol_ma10.items():
                            factor_data[(pd.Timestamp(date), stock)] = val
                        print(f'[VOL10] 股票 {stock} 因子计算完成，有效数据点: {vol_ma10.notna().sum()}')
                    else:
                        print(f'[VOL10] 股票 {stock} 无volume数据或数据为空')
                
                print(f'[VOL10] 所有股票处理完成，总因子数据点: {len(factor_data)}')
                if factor_data:
                    factor_series = pd.Series(factor_data)
                    factor_series.index.names = ['date', 'asset']
                    elapsed = time.time() - start_time
                    print(f'[get_single_factor完成] 自定义因子 {factor_name} - 耗时: {elapsed:.2f}秒')
                    return factor_series
            
            elif factor_name == 'single_day_VPT_12':
                factor_data = {}
                print(f'[VPT12] 开始计算single_day_VPT_12因子，股票数量: {len(self._get_stocks_for_calculation())}')
                for i, stock in enumerate(self._get_stocks_for_calculation()):
                    stock_start = time.time()
                    print(f'[VPT12] 拉取股票 {stock} 数据 ({i+1}/{len(self._get_stocks_for_calculation())}) - {time.strftime("%H:%M:%S")}')
                    
                    df = data.load_oss_complex_stocks([stock], start=self.cfg.START, end=self.cfg.END, fields='all')
                    
                    stock_elapsed = time.time() - stock_start
                    print(f'[VPT12] 股票 {stock} 数据拉取完成 - 耗时: {stock_elapsed:.2f}秒')
                    
                    if isinstance(df, dict) and 'close' in df and 'volume' in df and not df['close'].empty and not df['volume'].empty:
                        close = df['close'][stock]
                        vol = df['volume'][stock]
                        ret = close.pct_change()
                        vpt = (ret * vol).rolling(12).sum()
                        for date, val in vpt.items():
                            factor_data[(pd.Timestamp(date), stock)] = val
                        print(f'[VPT12] 股票 {stock} 因子计算完成，有效数据点: {vpt.notna().sum()}')
                    else:
                        print(f'[VPT12] 股票 {stock} 缺少close或volume数据或数据为空')
                
                print(f'[VPT12] 所有股票处理完成，总因子数据点: {len(factor_data)}')
                if factor_data:
                    factor_series = pd.Series(factor_data)
                    factor_series.index.names = ['date', 'asset']
                    elapsed = time.time() - start_time
                    print(f'[get_single_factor完成] 自定义因子 {factor_name} - 耗时: {elapsed:.2f}秒')
                    return factor_series
            
            # 尝试从 OSS 读取因子
            else:
                # 如果是 TALIB_ 开头的因子，直接跳到实时计算
                if factor_name.startswith('TALIB_'):
                    print(f'[TALIB因子] {factor_name} 直接使用实时计算 - {time.strftime("%H:%M:%S")}')
                else:
                    oss_start = time.time()
                    print(f'[OSS因子] 尝试从 OSS 读取因子: {factor_name} - {time.strftime("%H:%M:%S")}')
                    
                    try:
                        factor_series = data.factor_for_al(
                            codes=self._get_stocks_for_calculation(),
                            start_date=self.cfg.START,
                            end_date=self.cfg.END,
                            factor_name=factor_name
                        )
                        
                        oss_elapsed = time.time() - oss_start
                        print(f'[OSS因子] OSS读取完成 - 耗时: {oss_elapsed:.2f}秒')
                        
                        if factor_series is not None and len(factor_series) > 0:
                            print(f'✓ 成功从 OSS 读取因子 {factor_name}，共 {len(factor_series)} 条数据')
                            elapsed = time.time() - start_time
                            print(f'[get_single_factor完成] OSS因子 {factor_name} - 耗时: {elapsed:.2f}秒')
                            return factor_series
                        else:
                            print(f'⚠️  因子 {factor_name} 在 OSS 中无数据，尝试实时计算')
                            
                    except Exception as oss_e:
                        print(f'❌ 从 OSS 读取因子 {factor_name} 失败: {oss_e}，尝试实时计算')
                
                # 尝试使用计算器计算因子（TALIB_ 开头的因子或 OSS 失败的情况）
                try:
                    from factor_calculator import create_factor_calculator
                    calculator = create_factor_calculator(factor_name=factor_name)
                    
                    factor_data = {}
                    stocks = self._get_stocks_for_calculation()
                    print(f'[计算因子] 开始计算 {factor_name}，股票数量: {len(stocks)}')
                    
                    for i, stock in enumerate(stocks):
                        stock_start = time.time()
                        
                        try:
                            stock_series = calculator.calculate(stock, self.cfg.START, self.cfg.END)
                            if stock_series is not None and len(stock_series) > 0:
                                for date, val in stock_series.items():
                                    factor_data[(pd.Timestamp(date), stock)] = val
                            else:
                                pass  # 静默跳过无数据股票
                        except Exception as e:
                            print(f'[计算因子] 股票 {stock} 计算失败: {e}')  # 只打印失败的
                        
                        stock_elapsed = time.time() - stock_start
                    
                    if factor_data:
                        factor_series = pd.Series(factor_data)
                        factor_series.index.names = ['date', 'asset']
                        elapsed = time.time() - start_time
                        print(f'[get_single_factor完成] 计算因子 {factor_name} - 耗时: {elapsed:.2f}秒')
                        return factor_series
                    else:
                        print(f'⚠️  计算因子 {factor_name} 无有效数据')
                        return None
                        
                except Exception as calc_e:
                    print(f'❌ 计算因子 {factor_name} 失败: {calc_e}')
                    return None
        except Exception as e:
            elapsed = time.time() - start_time
            print(f'[get_single_factor失败] 因子 {factor_name} 计算失败：{e} - 耗时: {elapsed:.2f}秒')
            return None
        
        elapsed = time.time() - start_time
        print(f'[get_single_factor完成] 因子 {factor_name} 无数据 - 耗时: {elapsed:.2f}秒')
        return None
    
    def run_single_factor(self, factor_name, plot=False):
        """
        运行单个因子的检验
        
        Args:
            factor_name: 因子名称
            plot: 是否画图，默认 False
            
        Returns:
            list: FactorTestResult 列表
        """
        import time
        start_time = time.time()
        print(f'[开始] 因子 {factor_name} 检验 - {time.strftime("%H:%M:%S")}')
        
        # 只在第一次调用时加载股票和行业数据
        if not hasattr(self, '_stocks_loaded'):
            print(f'[步骤1] 加载股票池 - {time.strftime("%H:%M:%S")}')
            step_start = time.time()
            
            # 如果使用文件因子，需要先调用get_factors()的逻辑来获取股票列表和日期范围
            # 因为get_single_factor()不会调用get_factors()，所以需要手动处理
            if self.custom_factors:
                # 使用文件因子，从文件因子中获取股票列表和日期范围
                file_date_ranges = []
                file_stocks_list = []
                for calc in self.custom_factors.values():
                    if isinstance(calc, FileFactorCalculator):
                        file_start, file_end = calc.get_file_date_range()
                        if file_start and file_end:
                            file_date_ranges.append((file_start, file_end))
                        # 从文件因子中获取股票列表
                        stocks = calc.get_file_stocks()
                        if stocks:
                            file_stocks_list.extend(stocks)
                
                # 更新日期范围
                if file_date_ranges:
                    min_start = max(r[0] for r in file_date_ranges)
                    max_end = min(r[1] for r in file_date_ranges)
                    if min_start <= max_end:
                        self.cfg.START = min_start.strftime('%Y-%m-%d')
                        self.cfg.END = max_end.strftime('%Y-%m-%d')
                        print(f'[文件因子] 使用因子文件日期范围: {self.cfg.START} 到 {self.cfg.END}')
                
                # 获取股票列表
                if file_stocks_list:
                    self.stocks = sorted(list(set(file_stocks_list)))
                    print(f'[文件因子] 从因子文件中获取到 {len(self.stocks)} 只股票')
                elif not self.stocks:
                    # 如果文件因子中没有股票列表，尝试从配置获取
                    self.get_stocks()
            else:
                # 没有文件因子，先获取股票列表
                if not self.stocks:
                    self.get_stocks()
            
            # 如果股票列表仍然为空，给出警告并返回
            if not self.stocks:
                print(f'⚠️  警告：股票列表为空，无法进行因子验证')
                return []
            
            print(f'[步骤1完成] 股票池加载耗时: {time.time() - step_start:.2f}秒')
            
            print(f'[步骤2] 加载价格数据 - {time.strftime("%H:%M:%S")}')
            step_start = time.time()
            self._price = self.get_price()
            print(f'[步骤2完成] 价格数据加载耗时: {time.time() - step_start:.2f}秒')
            
            print(f'[步骤3] 加载行业数据 - {time.strftime("%H:%M:%S")}')
            step_start = time.time()
            self._industry = self.get_industry_series()
            print(f'[步骤3完成] 行业数据加载耗时: {time.time() - step_start:.2f}秒')
            
            self._stocks_loaded = True
        
        # 获取单个因子
        print(f'[步骤4] 计算因子 {factor_name} - {time.strftime("%H:%M:%S")}')
        step_start = time.time()
        factor_series = self.get_single_factor(factor_name)
        print(f'[步骤4完成] 因子计算耗时: {time.time() - step_start:.2f}秒')
        
        if factor_series is None or len(factor_series) == 0:
            print(f'⚠️  因子 {factor_name} 无数据')
            return []
        
        # 处理因子
        print(f'[步骤5] Alphalens因子清理 - {time.strftime("%H:%M:%S")}')
        step_start = time.time()
        
        # 清理因子数据，移除NaN值
        factor_clean = factor_series.dropna()
        print(f'[步骤5] 清理前因子数据点: {len(factor_series)}, 清理后: {len(factor_clean)}')
        
        # 检查有效数据比例
        if len(factor_series) == 0:
            print(f'⚠️  因子 {factor_name} 无任何数据')
            return []
        
        valid_ratio = len(factor_clean) / len(factor_series)
        print(f'[步骤5] 有效数据比例: {valid_ratio:.1%}')
        
        # 如果有效数据太少，可能是技术指标预热期问题
        if len(factor_clean) == 0:
            print(f'⚠️  因子 {factor_name} 清理后无有效数据')
            print(f'   可能原因：技术指标需要预热期，当前数据长度不足')
            print(f'   建议：增加历史数据长度或选择不同的因子')
            return []
        
        # 如果有效数据比例太低，给出警告但继续处理
        if valid_ratio < 0.1:  # 少于10%的有效数据
            print(f'⚠️  警告：因子 {factor_name} 有效数据比例仅为 {valid_ratio:.1%}')
            print(f'   可能原因：技术指标预热期较长，建议检查数据时间范围')
        
        try:
            clean = al.utils.get_clean_factor_and_forward_returns(
                factor_clean,
                prices=self._price,
                groupby=self._industry,
                quantiles=self.cfg.QUANTILES,
                periods=self.cfg.PERIODS,
                max_loss=1.0  # 允许最多丢弃 100% 的数据
            )
            print(f'[步骤5完成] Alphalens清理耗时: {time.time() - step_start:.2f}秒')
        except Exception as e:
            print(f'[步骤5失败] Alphalens清理失败: {e}')
            print(f'[步骤5失败] 尝试使用更简单的处理方式')
            
            # 使用更简单的处理方式
            clean = (factor_clean, None, None)
            print(f'[步骤5完成] 简单处理耗时: {time.time() - step_start:.2f}秒')
        
        # 处理并返回结果
        print(f'[步骤6] 处理因子结果 - {time.strftime("%H:%M:%S")}')
        step_start = time.time()
        results = self._process_single_factor(factor_name, clean, plot)
        print(f'[步骤6完成] 结果处理耗时: {time.time() - step_start:.2f}秒')
        
        total_time = time.time() - start_time
        print(f'[完成] 因子 {factor_name} 总耗时: {total_time:.2f}秒 - {time.strftime("%H:%M:%S")}')
        return results
    
    def _process_single_factor(self, factor_name, clean, plot=False):
        """
        处理单个因子的打分和 tear-sheet
        
        Args:
            factor_name: 因子名称
            clean: alphalens 清理后的因子数据
            plot: 是否画图
            
        Returns:
            list: FactorTestResult 列表
        """
        results = []
        
        if plot:
            try:
                # 导入 matplotlib
                import matplotlib
                import matplotlib.pyplot as plt
                
                # 关键：确保使用交互式 backend（alphalens 需要）
                # 如果当前是 Agg backend，临时切换到交互式 backend
                current_backend = matplotlib.get_backend()
                if current_backend == 'Agg':
                    import sys
                    if sys.platform == 'darwin':
                        matplotlib.use('macosx', force=True)
                    elif sys.platform.startswith('linux'):
                        matplotlib.use('TkAgg', force=True)
                    else:
                        matplotlib.use('TkAgg', force=True)
                    print(f'  切换 backend 从 {current_backend} 到 {matplotlib.get_backend()} 以生成图表')
                
                # 清除之前的图表（避免内存泄漏）
                plt.close('all')
                
                # 关键：拦截 plt.show() 调用，在调用时立即保存图表
                # alphalens 内部调用 plt.show() 时，图表可能被清除
                # 解决方案：在 plt.show() 调用时立即保存所有图表到临时位置
                saved_fig_nums = set()
                saved_fig_objects = {}  # 保存 figure 对象的引用
                original_show = plt.show
                
                def save_and_mock_show(*args, **kwargs):
                    # 在 plt.show() 调用时，立即保存所有未保存的图表
                    # 注意：alphalens 可能生成多个图表，第一个可能是空的
                    current_figs = plt.get_fignums()
                    for fig_num in current_figs:
                        if fig_num not in saved_fig_nums:
                            try:
                                fig = plt.figure(fig_num)
                                # 确保图表完全渲染
                                fig.canvas.draw()
                                # 保存 figure 对象的引用（避免被清除）
                                saved_fig_objects[fig_num] = fig
                                saved_fig_nums.add(fig_num)
                            except Exception as e:
                                print(f'  保存图表 {fig_num} 时出错: {e}')
                    # 不实际显示
                    pass
                
                plt.show = save_and_mock_show
                
                try:
                    # 生成 tear-sheet（使用交互式 backend，确保正确生成）
                    # 注意：group_neutral=True 和 by_group=True 可能导致错误，如果出错会回退到 False
                    try:
                        al.tears.create_full_tear_sheet(clean,
                                                        long_short=True,
                                                        group_neutral=True,
                                                        by_group=True)
                    except Exception as e:
                        # 如果 group_neutral/by_group 出错，尝试不使用这些选项
                        print(f'  使用 group_neutral/by_group 时出错: {e}')
                        print(f'  尝试使用更简单的配置...')
                        al.tears.create_full_tear_sheet(clean,
                                                        long_short=True,
                                                        group_neutral=False,
                                                        by_group=False)
                except Exception as e:
                    # 即使 tear-sheet 生成出错，也可能已经生成了一些图表
                    print(f'  tear-sheet 生成时出错: {e}')
                    print(f'  但已捕获的图表: {list(saved_fig_objects.keys())}')
                finally:
                    # 重要：在恢复 plt.show 之前，确保所有图表都被捕获
                    # 再次检查是否有新的图表出现
                    current_figs = plt.get_fignums()
                    for fig_num in current_figs:
                        if fig_num not in saved_fig_nums:
                            try:
                                fig = saved_fig_objects[fig_num]
                                fig.canvas.draw()  # 强制渲染
                            except:
                                pass
                    
                    # 恢复 plt.show
                    plt.show = original_show
                    
                    # 重要：确保所有图表都被渲染（即使出错了）
                    # 使用保存的 figure 对象引用，确保图表不被清除
                    all_fig_nums = set(plt.get_fignums()) | set(saved_fig_objects.keys())
                    for fig_num in all_fig_nums:
                        try:
                            if fig_num in saved_fig_objects:
                                fig = saved_fig_objects[fig_num]
                            else:
                                fig = plt.figure(fig_num)
                            fig.canvas.draw()  # 强制渲染
                        except:
                            pass
                
                # 检查生成的图表数量（包括保存的图表）
                current_figs = plt.get_fignums()
                all_figs = set(current_figs) | set(saved_fig_objects.keys())
                figs_count = len(all_figs)
                
                if figs_count > 0:
                    print(f'✅ {factor_name} tear-sheet 生成成功，共 {figs_count} 个图表')
                else:
                    print(f'⚠️  {factor_name} tear-sheet 未生成任何图表')
                
                # 提示：alphalens 可能生成多个图表，第一个可能是空的
                # 调用方需要保存所有图表，不要只保存第一个
                
                # 重要：将保存的图表对象保存到实例变量，供调用方使用
                # 注意：必须保存字典的副本，避免引用被修改
                self._saved_fig_objects = dict(saved_fig_objects)
                
                # 注意：图表已经生成，但可能不会显示（取决于 backend）
                # 如果需要保存图片，应该在调用方使用 plt.savefig() 或 plt.get_fignums()
            except Exception as e:
                print(f'⚠️  {factor_name} tear-sheet 异常：{e}')
                import traceback
                traceback.print_exc()
        else:
            print(f'📊 {factor_name} 跳过 tear-sheet（未启用画图）')

        # 对每个调仓周期分别打分
        for p in self.cfg.PERIODS:
            # 检查 clean 数据格式，如果是失败时的元组格式，跳过打分
            if isinstance(clean, tuple) and len(clean) == 3 and clean[1] is None and clean[2] is None:
                print(f'⚠️  {factor_name} 周期{p}天 Alphalens清理失败，跳过打分')
                continue
            
            result = quick_score(factor_name, clean, p, self.cfg)
            
            # 检查是否返回了 None（数据不足或计算失败）
            if result[0] is None:
                print(f'⚠️  {factor_name} 周期{p}天 数据不足，跳过')
                continue
            
            scores, level, ic_series, ret_series, top_turnover = result
            print(f'\n📊 {factor_name} 周期{p}天 打分结果：{level}')
            for k, (v, ok) in scores.items():
                print(f'   {k:12s}: {v:7.3f}  {"✅" if ok else "❌"}')

            # ★ 滚动监控 + 红黄绿灯判定（变量现成）
            latest = rolling_monitor(factor_name, ic_series, ret_series, p, self.cfg)

            roll_ir   = latest.roll_ir
            win_rate  = (ic_series > 0).rolling(self.cfg.ROLL_WIN).mean().iloc[-1]
            std_ret   = ret_series.std()
            net_ir    = np.nan if std_ret == 0 or np.isnan(std_ret) else \
                        (ret_series.mean() - top_turnover * 0.003 * 2) / std_ret * np.sqrt(252)
            ic_recent = ic_series.dropna().iloc[-5:]
            q5q1_shrp = np.nan if std_ret == 0 else ret_series.mean() / std_ret * np.sqrt(252)

            flag = '🔴 dead'   if (roll_ir < 0.2 or win_rate < 0.5 or net_ir < 0 or
                                   (ic_recent < 0).sum() >= 5 or q5q1_shrp < 0) else \
                   '🟡 warning' if (roll_ir < 0.3 or win_rate < 0.52 or net_ir < 0.2 or
                                   (ic_recent < 0).sum() >= 3 or q5q1_shrp < 0.5) else \
                   '🟢 alive'

            print(f'   {"滚IC":12s}: {latest.roll_ic:8.3f}  '
                  f'{"滚IR":12s}: {latest.roll_ir:8.3f}  '
                  f'{"滚t":12s}: {latest.roll_t:8.2f}  '
                  f'{"Top波动":12s}: {latest.top_std:8.1%}  '
                  f'{"负日率":12s}: {latest.neg_day:8.1%}  '
                  f'状态:{flag}')
            
            # 创建结果对象
            result = FactorTestResult()
            result.factor_name = factor_name
            result.period = p
            result.scores = scores
            result.level = level
            result.ic_series = ic_series
            result.ret_series = ret_series
            result.top_turnover = top_turnover
            result.rolling_monitor = latest
            result.status_flag = flag
            result.clean_data = clean
            results.append(result)
        
        return results


def get_stock_list(stock_pool: str, max_stocks: Optional[int] = None) -> List[str]:
    """
    获取股票列表

    Args:
        stock_pool: 股票池标识
        max_stocks: 最大股票数量

    Returns:
        股票代码列表
    """
    try:
        if stock_pool == 'stock':
            # 全市场股票 - 这里需要实现获取全市场股票的逻辑
            # 暂时使用沪深300作为示例
            stocks = data.get_index_stocks('000300')
        elif stock_pool == 'small':
            # 小盘股（市值较小的股票）
            # 暂时使用中证500作为小盘股代表
            stocks = data.get_index_stocks('000905')[:500] if len(data.get_index_stocks('000905')) > 500 else data.get_index_stocks('000905')
        else:
            # 指数成分股
            stocks = data.get_index_stocks(stock_pool)

        if max_stocks and len(stocks) > max_stocks:
            stocks = stocks[:max_stocks]

        print(f"获取到 {len(stocks)} 只股票")
        return stocks

    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return []


def generate_single_factor(factor_name: str, stock_codes: List[str],
                          start_date: str, end_date: str,
                          factor_file: Optional[str] = None,
                          factor_dir: Optional[str] = None) -> pd.DataFrame:
    """
    生成单个因子的数据

    Args:
        factor_name: 因子名称
        stock_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        factor_file: 因子文件路径
        factor_dir: 因子文件目录

    Returns:
        包含因子数据的DataFrame (date, code, factor_value)
    """
    print(f"\n开始生成因子: {factor_name}")

    # 创建因子计算器
    try:
        if factor_file and factor_name:
            # 从文件加载因子
            calc = create_factor_calculator(
                file_path=factor_file,
                factor_name=factor_name
            )
        elif factor_dir and factor_name:
            # 从目录查找因子文件
            calc = create_factor_calculator(
                factor_name=factor_name,
                factor_dir=factor_dir
            )
        else:
            # 使用内置因子
            calc = create_factor_calculator(factor_name=factor_name)

        print(f"✓ 创建因子计算器成功: {type(calc).__name__}")

    except Exception as e:
        print(f"❌ 创建因子计算器失败: {e}")
        return pd.DataFrame()

    # 为每只股票计算因子
    all_factor_data = []

    for i, stock_code in enumerate(stock_codes):
        if (i + 1) % 50 == 0:
            print(f"  处理进度: {i+1}/{len(stock_codes)} 股票")

        try:
            # 计算因子值
            factor_series = calc.calculate(stock_code, start_date, end_date)

            if not factor_series.empty:
                # 删除NaN值，只保留有效的数据点
                factor_series = factor_series.dropna()

                if not factor_series.empty:
                    # 转换为标准格式
                    stock_data = pd.DataFrame({
                        'date': factor_series.index,
                        'code': str(stock_code).zfill(6),  # 标准化为6位数字
                        'factor_value': factor_series.values
                    })
                    # 确保code列是字符串类型
                    stock_data['code'] = stock_data['code'].astype(str)
                    all_factor_data.append(stock_data)

        except Exception as e:
            print(f"  ⚠️  计算股票 {stock_code} 因子失败: {e}")
            continue

    # 合并所有股票的数据
    if all_factor_data:
        result_df = pd.concat(all_factor_data, ignore_index=True)

        # 排序并重置索引
        result_df = result_df.sort_values(['date', 'code']).reset_index(drop=True)

        print(f"✓ 因子 {factor_name} 生成完成，共 {len(result_df)} 条记录")
        return result_df
    else:
        print(f"❌ 因子 {factor_name} 生成失败，无有效数据")
        return pd.DataFrame()


def save_factor_data(factor_name: str, factor_data: pd.DataFrame,
                    output_dir: str, start_date: str, end_date: str,
                    overwrite: bool = False) -> bool:
    """
    保存因子数据到文件

    Args:
        factor_name: 因子名称
        factor_data: 因子数据DataFrame
        output_dir: 输出目录
        overwrite: 是否覆盖现有文件

    Returns:
        保存是否成功
    """
    if factor_data.empty:
        print(f"⚠️  因子 {factor_name} 数据为空，跳过保存")
        return False

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 生成文件名
    filename = f"{factor_name}_{start_date}_{end_date}.csv"
    filepath = output_path / filename

    # 检查文件是否已存在
    if filepath.exists() and not overwrite:
        print(f"⚠️  文件已存在: {filepath}，使用 --overwrite 覆盖")
        return False

    try:
        # 保存为CSV
        factor_data.to_csv(filepath, index=False, float_format='%.6f')

        # 显示统计信息
        print(f"✓ 因子 {factor_name} 已保存: {filepath}")
        print(f"  数据量: {len(factor_data)} 条记录")
        print(f"  股票数: {factor_data['code'].nunique()}")
        print(f"  日期范围: {factor_data['date'].min()} ~ {factor_data['date'].max()}")

        return True

    except Exception as e:
        print(f"❌ 保存因子 {factor_name} 失败: {e}")
        return False


def export_data(
    codes: List[str],
    factors: List[str],
    start_date: str,
    end_date: str,
    output_dir: str,
    stock_pool: str = 'stock',
    max_stocks: Optional[int] = None,
    factor_file: Optional[str] = None,
    factor_dir: Optional[str] = None,
    overwrite: bool = False
) -> str:
    """
    导出因子数据，使用generate_factors.py的实现

    Args:
        codes: 股票代码列表（如果为空，使用stock_pool获取）
        factors: 因子名称列表
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        output_dir: 输出目录
        stock_pool: 股票池类型，默认 'stock'
        max_stocks: 最大股票数量限制
        factor_file: 因子文件路径
        factor_dir: 因子文件目录
        overwrite: 是否覆盖现有文件

    Returns:
        str: 输出文件路径
    """
    import os
    from datetime import datetime
    from pathlib import Path
    from pathlib import Path

    print(f"\n{'='*60}")
    print(f"导出因子数据")
    print(f"{'='*60}")
    print(f"股票池: {stock_pool}")
    print(f"股票数量: {len(codes) if codes else '自动获取'}")
    print(f"因子列表: {factors}")
    print(f"日期范围: {start_date} ~ {end_date}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}\n")

    # 获取股票列表
    if not codes:
        codes = get_stock_list(stock_pool, max_stocks)
        if not codes:
            print("❌ 无法获取股票列表")
            return None

    # 为技术指标计算扩展时间范围（向前推3个月以确保有足够数据）
    import pandas as pd
    extended_start = pd.to_datetime(start_date) - pd.DateOffset(months=3)
    extended_start_str = extended_start.strftime('%Y-%m-%d')
    print(f"为技术指标计算扩展时间范围: {extended_start_str} ~ {end_date} (原始: {start_date} ~ {end_date})")

    # 生成因子数据
    factor_results = []

    for factor_name in factors:
        try:
            # 生成单个因子数据
            factor_data = generate_single_factor(
                factor_name=factor_name,
                stock_codes=codes,
                start_date=extended_start_str,  # 使用扩展的时间范围
                end_date=end_date,
                factor_file=factor_file if factor_name == factors[0] else None,  # 只对第一个因子使用文件
                factor_dir=factor_dir
            )

            # 保存因子数据
            success = save_factor_data(
                factor_name=factor_name,
                factor_data=factor_data,
                output_dir=output_dir,
                start_date=start_date,
                end_date=end_date,
                overwrite=overwrite
            )

            factor_results.append({
                'factor_name': factor_name,
                'success': success,
                'records': len(factor_data) if success else 0
            })

        except Exception as e:
            print(f"❌ 生成因子 {factor_name} 时发生错误: {e}")
            factor_results.append({
                'factor_name': factor_name,
                'success': False,
                'records': 0
            })

    # 统计结果
    successful_count = sum(1 for r in factor_results if r['success'])
    total_records = sum(r['records'] for r in factor_results)

    print(f"\n✓ 导出完成: {successful_count}/{len(factors)} 个因子成功")
    print(f"  总数据量: {total_records} 条记录")
    print(f"  输出目录: {output_dir}")

    return output_dir


def export_formatted_csv(
    codes: List[str],
    start_date: str,
    end_date: str,
    output_dir: str,
    factors: List[str] = None,   # 新：允许指定因子子集，None或[]时为全量
    industry_default: str = "Unknown"
) -> str:
    """
    只导出基础行情+原生OSS因子字段：
    date, stock, open, high, low, close, volume, amount, mkt_cap, industry, <全量或指定因子>
    不做任何派生/二次计算。
    """
    print(f"\n{'='*60}")
    print(f"导出原始行情+原生因子字段数据...")
    print(f"股票数量: {len(codes)}  日期: {start_date} ~ {end_date}")
    print(f"因子名: {'ALL' if not factors else factors}")
    print(f"{'='*60}\n")

    # --------- 行情部分 ---------
    # 为价格数据也使用扩展时间范围（与因子计算一致）
    extended_start = pd.to_datetime(start_date) - pd.DateOffset(months=3)
    extended_start_str = extended_start.strftime('%Y-%m-%d')
    
    price_dict = data.load_oss_complex_stocks(
        codes=codes,
        start=extended_start_str,  # 使用扩展时间范围
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
    # 简单市值计算
    if 'close' in merged.columns and 'outstanding_share' in merged.columns:
        merged['mkt_cap'] = merged['close'] * merged['outstanding_share']
    else:
        merged['mkt_cap'] = pd.NA
    # 行业与概念填充
    try:
        code_list = merged['stock'].dropna().astype(str).unique().tolist()
        ind_map = data.get_industry_category(code_list) if code_list else {}
        cpt_map = data.get_concept_categories(code_list) if code_list else {}
    except Exception:
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

    # --------- 因子部分 ---------
    df_factors = None
    if factors and len(factors) > 0:
        # 从已导出的因子文件中读取数据
        factor_frames = []
        all_stock_codes = set()  # 收集所有股票代码
        
        for factor_name in factors:
            factor_file = os.path.join(output_dir, f"{factor_name}_{start_date}_{end_date}.csv")
            if os.path.exists(factor_file):
                try:
                    df = pd.read_csv(factor_file, dtype={'code': str})
                    df['date'] = pd.to_datetime(df['date'])
                    # 收集股票代码
                    all_stock_codes.update(df['code'].dropna().unique())
                    # 重命名列以匹配期望的格式
                    df = df.rename(columns={'code': 'stock', 'factor_value': factor_name})
                    # 只保留需要的列
                    df = df[['date', 'stock', factor_name]]
                    factor_frames.append(df)
                except Exception as e:
                    print(f"⚠️  读取因子文件 {factor_file} 失败: {e}")
                    continue
        
        if factor_frames:
            # 合并所有因子数据
            df_factors = factor_frames[0]
            for df in factor_frames[1:]:
                df_factors = df_factors.merge(df, on=['date', 'stock'], how='outer')
            
            # 设置MultiIndex
            df_factors = df_factors.set_index(['date', 'stock'])
            
            # 更新codes为所有因子相关的股票
            codes = list(all_stock_codes)
    
    if df_factors is not None and not df_factors.empty:
        # df_factors: MultiIndex(date, code)
        df_factors = df_factors.reset_index()
        # 'code'字段源码取名与行情对齐
        df_factors.rename(columns={"code":"stock"}, inplace=True)
        # 统一两侧股票代码格式为6位数字，避免后缀/前缀不一致导致合并失败
        def _to_six_digit(s: pd.Series) -> pd.Series:
            s = s.astype(str).str.upper()
            s = (s.str.replace('.XSHG','', regex=False)
                   .str.replace('.XSHE','', regex=False)
                   .str.replace('.XBJ','', regex=False))
            extracted = s.str.extract(r'(\d{6})')[0]
            s = extracted.where(extracted.notna(), s)
            return s
        
        merged['stock'] = _to_six_digit(merged['stock'])
        if df_factors is not None:
            # 重新标准化因子数据的股票代码
            df_factors_reset = df_factors.reset_index()
            df_factors_reset['stock'] = _to_six_digit(df_factors_reset['stock'])
            df_factors = df_factors_reset.set_index(['date', 'stock'])
        merged = merged.merge(df_factors, on=["date","stock"], how="outer")
        # 确保基础字段和所有原生因子顺序
        base_cols = ['date', 'stock', 'open', 'high', 'low', 'close', 'volume', 'amount', 'mkt_cap', 'industry', 'concepts']
        factor_cols = [c for c in df_factors.columns if c not in base_cols and c not in ('date','stock')]
        # 若指定factors有顺序则保持，否则用读取到的原始顺序
        if factors and len(factors)>0:
            factor_cols = [c for c in factors if c in factor_cols]
        final_cols = base_cols + factor_cols
    else:
        print("⚠️  无任何因子可导出，仅输出基础行情字段")
        base_cols = ['date', 'stock', 'open', 'high', 'low', 'close', 'volume', 'amount', 'mkt_cap', 'industry']
        final_cols = base_cols
    # 补空
    for c in final_cols:
        if c not in merged.columns:
            merged[c] = pd.NA
    out_df = merged[final_cols].sort_values(['date','stock']).reset_index(drop=True)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'formatted_data.csv')
    out_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ 原始宽表已导出: {output_file}")
    print(f"  行数: {len(out_df)}  股票: {out_df['stock'].nunique()}  日期: {out_df['date'].min()} ~ {out_df['date'].max()}")
    print(f"  导出因子: {factor_cols if df_factors is not None and not df_factors.empty else '无'}")
    return output_file


# -----------------------------------------------------------
# 4. 运行
# -----------------------------------------------------------
def main():
    """主函数"""
    args = parse_args()
    cfg = CFG(args)
    
    # 打印配置信息
    print("=" * 60)
    print("因子检验配置")
    print("=" * 60)
    print(f"回测区间: {cfg.START} ~ {cfg.END}")
    print(f"股票池: {cfg.STOCK_POOL}")
    print(f"因子列表: {cfg.FACTORS}")
    if cfg.FACTOR_DIR:
        print(f"因子文件目录: {cfg.FACTOR_DIR}")
    print(f"分位数: {cfg.QUANTILES}")
    print(f"调仓周期: {cfg.PERIODS}")
    print(f"滚动窗口: {cfg.ROLL_WIN} 天")
    print(f"监控文件: {cfg.MONITOR_CSV}")
    print("=" * 60)
    print()
    
    # 如果提供了因子文件目录，创建因子计算器
    custom_factors = None
    if cfg.FACTOR_DIR:
        print(f"[配置] 从因子文件目录加载因子: {cfg.FACTOR_DIR}")
        custom_factors = {}
        for factor_name in cfg.FACTORS:
            try:
                calc = create_factor_calculator(
                    factor_name=factor_name,
                    factor_dir=cfg.FACTOR_DIR
                )
                custom_factors[factor_name] = calc
                print(f"  ✓ {factor_name}: 已创建因子计算器")
            except Exception as e:
                print(f"  ✗ {factor_name}: 创建失败 - {e}")
        if not custom_factors:
            print("⚠️  警告: 未成功创建任何因子计算器，将使用默认方法")
            custom_factors = None
        print()
    
    # 运行因子检验
    tester = FactorTester(cfg, custom_factors=custom_factors)
    tester.run()


if __name__ == '__main__':
    main()

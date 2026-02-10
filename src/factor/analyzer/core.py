"""
因子分析核心方法

从旧的 factor.py 中复制的核心方法：
- get_price(): 获取价格数据
- get_industry_series(): 获取行业分组序列
- run() 方法中的因子处理逻辑
- FactorAnalyzer: 因子分析器类
"""

import os
import sys
import tempfile
import pandas as pd
import numpy as np
import time
import re
from typing import List, Optional, Dict, Any


# --- 路径与依赖配置 ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FACTOR_DIR = os.path.dirname(CURRENT_DIR)
SRC_ROOT = os.path.dirname(FACTOR_DIR)
REPO_ROOT = os.path.dirname(SRC_ROOT)

for path in (FACTOR_DIR, SRC_ROOT, REPO_ROOT):
    if path and path not in sys.path:
        sys.path.insert(0, path)

import alphalens as al
import data


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


class FactorAnalysisCore:
    """因子分析核心方法"""

    def __init__(self, cfg, output_dir=None):
        """
        初始化因子分析核心

        Args:
            cfg: 配置对象，包含 START, END, STOCK_POOL 等属性
            output_dir: 输出目录，用于保存图表等文件
        """
        self.cfg = cfg
        self.output_dir = output_dir
        self.stocks = []  # 股票列表，需要外部设置

    def _get_stocks_for_calculation(self):
        """获取用于计算的股票列表，根据配置决定是否限制数量"""
        if self.cfg.MAX_STOCKS is not None:
            # 限制股票数量用于测试
            stocks = self.stocks[:self.cfg.MAX_STOCKS]
            print(f'[配置] 限制股票数量为 {self.cfg.MAX_STOCKS} 只')
        else:
            stocks = self.stocks
        return stocks

    def get_price(self):
        """获取价格数据"""
        start_time = time.time()
        print(f'[get_price开始] 正在拉取行情... - {time.strftime("%H:%M:%S")}')

        remaining_stocks = self._get_stocks_for_calculation()
        failed_stocks = []

        while remaining_stocks:
            try:
                price = data.load_oss_stocks(
                    codes=remaining_stocks,
                    start=self.cfg.START,
                    end=self.cfg.END
                )
                price.index.name = 'date'
                elapsed = time.time() - start_time
                if failed_stocks:
                    print(f'[get_price完成] 耗时: {elapsed:.2f}秒 - 已跳过 {len(failed_stocks)} 只股票: {", ".join(failed_stocks)}')
                else:
                    print(f'[get_price完成] 耗时: {elapsed:.2f}秒')
                return price
            except Exception as e:
                # 尝试从错误信息中解析失败的股票代码
                err_msg = str(e)
                failed_code = None

                patterns = [
                    r'股票\s*(\d{6})',
                    r's[hz](\d{6})',
                    r'(\d{6})'
                ]
                for pattern in patterns:
                    match = re.search(pattern, err_msg, re.IGNORECASE)
                    if match:
                        failed_code = match.group(1)
                        break

                # 如果能确定失败的股票，就移除后重试
                if failed_code and failed_code in remaining_stocks:
                    print(f'[get_price警告] 股票 {failed_code} 数据拉取失败，已跳过。错误: {err_msg.splitlines()[0]}')
                    failed_stocks.append(failed_code)
                    remaining_stocks = [code for code in remaining_stocks if code != failed_code]
                    continue

                # 无法识别具体股票或已无剩余股票，终止
                elapsed = time.time() - start_time
                print(f'[get_price失败] 耗时: {elapsed:.2f}秒 - 错误: {e}')
                return pd.DataFrame()

        elapsed = time.time() - start_time
        print(f'[get_price失败] 耗时: {elapsed:.2f}秒 - 所有股票数据均拉取失败')
        return pd.DataFrame()

    def get_industry_series(self):
        """获取行业/概念分组序列"""
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

    def process_factors_with_alphalens(self, factors: Dict[str, pd.Series],
                                     price: pd.DataFrame,
                                     industry: pd.Series,
                                     plot: bool = False) -> List[Dict[str, Any]]:
        """
        使用 Alphalens 处理因子数据

        Args:
            factors: 因子字典 {factor_name: factor_series}
            price: 价格数据
            industry: 行业分组数据
            plot: 是否生成图表

        Returns:
            处理后的结果列表
        """
        results = {}
        for f, factor_series in factors.items():
            try:
                clean = al.utils.get_clean_factor_and_forward_returns(
                    factor_series,
                    prices=price,
                    groupby=industry,
                    quantiles=self.cfg.QUANTILES,
                    periods=self.cfg.PERIODS,
                    max_loss=3
                )
                results[f] = clean
                print(f'✓ 因子 {f} 处理完成')
            except Exception as e:
                print(f'⚠️  因子 {f} 处理失败: {e}')
                continue

        # 存储所有检验结果
        all_results = []

        # 打分 & tear-sheet
        for f, clean in results.items():
            try:
                # 调用 _process_single_factor 方法处理单个因子
                factor_results = self._process_single_factor(f, clean, plot=plot)
                all_results.extend(factor_results)
                print(f'✓ 因子 {f} 分析完成')
            except Exception as e:
                print(f'⚠️  因子 {f} 分析失败: {e}')
                continue

        print('全部完成！')
        return all_results

    def _quick_score(self, factor_name, factor_data, period_days):
        """
        根据调仓周期动态调整阈值进行打分

        Args:
            factor_name: 因子名称
            factor_data: 因子数据
            period_days: 调仓周期（天）

        Returns:
            tuple: (scores, level, ic_series, ret_series, top_turnover) 或 None
        """
        try:
            from alphalens.performance import (
                factor_information_coefficient,
                factor_returns,
                quantile_turnover,
                mean_return_by_quantile
            )
            import pandas as pd
            import numpy as np

            # 1. IC
            ic = factor_information_coefficient(factor_data)
            if ic.empty or len(ic) == 0:
                print(f'⚠️  {factor_name} IC 数据为空')
                return None
            
            # 确保 ic 是数值类型
            ic = ic.apply(pd.to_numeric, errors='coerce')
            
            ic_mean = ic.mean().iloc[0]
            ic_std = ic.std().iloc[0]
            icir = ic_mean / ic_std if ic_std != 0 and not np.isnan(ic_std) else np.nan
            
        except Exception as e:
            print(f'⚠️  {factor_name} IC 计算失败: {e}')
            return None

        # 2. Top-Bottom 收益
        ret = factor_returns(factor_data)
        ret_col = ret.columns[self.cfg.PERIODS.index(period_days)]
        tb_ret = ret[ret_col]
        ann_ret = tb_ret.mean() * 252

        # 3. Top 分位数换手率
        quant_col = factor_data['factor_quantile']
        top_quantile = self.cfg.QUANTILES
        top = (quant_col[quant_col == top_quantile]
               .reset_index()
               .groupby('date')['asset']
               .apply(list))

        top_sets = top.apply(lambda x: set(x) if isinstance(x, list) else set())
        shift_sets = top.shift().apply(lambda x: set(x) if isinstance(x, list) else set())

        intersection = pd.Series(
            [len(a & b) for a, b in zip(top_sets, shift_sets)],
            index=top.index
        )

        denom = shift_sets.apply(len)
        turnover_rate = 1 - (intersection / denom)
        top_turnover = turnover_rate.dropna().mean()

        # 4. 单调性
        mean_ret, _ = mean_return_by_quantile(
            factor_data, by_group=False, demeaned=True)
        mean_col = mean_ret.columns[self.cfg.PERIODS.index(period_days)]
        
        if len(mean_ret.index) < 2:
            mono = np.nan
        else:
            actual_quantiles = len(mean_ret.index)
            mono = mean_ret[mean_col].corr(pd.Series(range(1, actual_quantiles + 1),
                                                     index=mean_ret.index))

        # 5. 简化的打分（基于原始代码的逻辑）
        scores = {
            'IC均值': (ic_mean, abs(ic_mean) > 0.03),
            'IR比率': (icir, icir > 0.8),
            '多空年化': (ann_ret, ann_ret > 0.1),
            '单调性': (mono, mono > 0.8),
            'Top换手率': (top_turnover, top_turnover < 0.6)
        }

        passed = sum(scores[k][1] for k in scores)
        level = '优秀' if passed >= 4 else ('良好' if passed >= 3 else '一般')

        return scores, level, ic.iloc[:, 0], tb_ret, top_turnover

    def _rolling_monitor(self, factor_name, ic_series, ret_series, period_days):
        """
        滚动监控因子表现
        
        Args:
            factor_name: 因子名称
            ic_series: IC序列
            ret_series: 收益序列
            period_days: 调仓周期
            
        Returns:
            滚动监控结果对象
        """
        class RollingMonitorResult:
            def __init__(self):
                self.roll_ic = np.nan
                self.roll_ir = np.nan
                self.roll_t = np.nan
                self.top_std = np.nan
                self.neg_day = np.nan
        
        result = RollingMonitorResult()
        
        try:
            # 计算滚动IC均值
            roll_win = getattr(self.cfg, 'ROLL_WIN', 20)  # 默认20天
            result.roll_ic = ic_series.rolling(roll_win).mean().iloc[-1]
            
            # 计算滚动IR
            roll_ic_std = ic_series.rolling(roll_win).std().iloc[-1]
            result.roll_ir = result.roll_ic / roll_ic_std if roll_ic_std != 0 else np.nan
            
            # 计算滚动t值
            n = len(ic_series.dropna())
            if n > 0:
                result.roll_t = result.roll_ic / (ic_series.std() / np.sqrt(n)) if ic_series.std() != 0 else np.nan
            
            # Top组合波动率
            result.top_std = ret_series.std()
            
            # 负收益日占比
            result.neg_day = (ret_series < 0).mean()
            
        except Exception as e:
            print(f'⚠️  {factor_name} 滚动监控计算失败: {e}')
        
        return result

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
                captured_fig_ids = set()
                captured_figures = []  # 按捕获顺序保存 figure 引用
                original_show = plt.show
                
                def save_and_mock_show(*args, **kwargs):
                    # 在 plt.show() 调用时，立即保存所有未保存的图表
                    # 注意：alphalens 可能生成多个图表，第一个可能是空的
                    current_figs = plt.get_fignums()
                    if getattr(self.cfg, 'DEBUG_PLOT', False):
                        print(f'  [调试] plt.show 捕获 {len(current_figs)} 个 figure: {list(current_figs)}')
                    for fig_num in current_figs:
                        try:
                            fig = plt.figure(fig_num)
                            # 确保图表完全渲染
                            fig.canvas.draw()
                            fig_id = id(fig)
                            if fig_id not in captured_fig_ids:
                                captured_figures.append(fig)
                                captured_fig_ids.add(fig_id)
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
                    print(f'  但已捕获的图表数量: {len(captured_figures)}')
                finally:
                    # 重要：在恢复 plt.show 之前，确保所有图表都被捕获
                    # 再次检查是否有新的图表出现
                    current_figs = plt.get_fignums()
                    for fig_num in current_figs:
                        try:
                            fig = plt.figure(fig_num)
                            fig_id = id(fig)
                            if fig_id not in captured_fig_ids:
                                fig.canvas.draw()
                                captured_figures.append(fig)
                                captured_fig_ids.add(fig_id)
                        except Exception:
                            pass
                    
                    # 恢复 plt.show
                    plt.show = original_show
                    
                    # 重要：确保所有图表都被渲染（即使出错了）
                    # 使用保存的 figure 对象引用，确保图表不被清除
                    current_figs = plt.get_fignums()
                    for fig_num in current_figs:
                        try:
                            fig = plt.figure(fig_num)
                            fig_id = id(fig)
                            if fig_id not in captured_fig_ids:
                                fig.canvas.draw()
                                captured_figures.append(fig)
                                captured_fig_ids.add(fig_id)
                        except Exception:
                            pass
                
                # 检查生成的图表数量（包括保存的图表）
                current_figs = plt.get_fignums()
                for fig_num in current_figs:
                    try:
                        fig = plt.figure(fig_num)
                        fig_id = id(fig)
                        if fig_id not in captured_fig_ids:
                            fig.canvas.draw()
                            captured_figures.append(fig)
                            captured_fig_ids.add(fig_id)
                    except Exception:
                        pass

                figs_count = len(captured_figures)
                
                if figs_count > 0:
                    print(f'✅ {factor_name} tear-sheet 生成成功，共 {figs_count} 个图表')
                    
                    # 保存所有图表为PNG文件
                    saved_count = 0
                    kept_images = []

                    def _analyze_image_content(path: str):
                        """返回 (是否接近空白, 非白像素占比, 像素标准差)。"""
                        try:
                            from PIL import Image
                            import numpy as np
                        except ImportError:
                            return False, None, None  # 如果 Pillow 不可用，保留图片

                        try:
                            with Image.open(path) as im:
                                arr = np.array(im.convert('RGB'))
                                total = arr.shape[0] * arr.shape[1]
                                if total == 0:
                                    return True, 0.0, 0.0

                                non_white = np.count_nonzero(np.any(arr < 245, axis=2))
                                non_white_ratio = non_white / total
                                luminance_std = float(arr.std())

                                is_blank = (non_white_ratio < 0.0005 and luminance_std < 2.0)
                                return is_blank, non_white_ratio, luminance_std
                        except Exception:
                            return False, None, None

                    for i, fig in enumerate(captured_figures, 1):
                        try:
                            # 确保图表完全渲染
                            fig.canvas.draw()

                            fd, tmp_path = tempfile.mkstemp(suffix='.png')
                            os.close(fd)

                            fig.savefig(
                                tmp_path,
                                dpi=200,
                                bbox_inches='tight',
                                facecolor='white',
                                edgecolor='none',
                                format='png'
                            )

                            file_exists = os.path.exists(tmp_path)
                            file_size = os.path.getsize(tmp_path) if file_exists else 0
                            is_blank = False
                            non_white_ratio = None
                            luminance_std = None

                            if file_exists:
                                is_blank, non_white_ratio, luminance_std = _analyze_image_content(tmp_path)

                            should_keep = file_exists and file_size > 1500 and (not is_blank)

                            if should_keep:
                                kept_images.append({
                                    'tmp_path': tmp_path,
                                    'file_size': file_size,
                                    'non_white_ratio': non_white_ratio,
                                    'luminance_std': luminance_std,
                                    'source_index': i
                                })
                            else:
                                if file_exists:
                                    os.remove(tmp_path)
                                extra = ''
                                if non_white_ratio is not None and luminance_std is not None:
                                    extra = f' | 非白像素占比={non_white_ratio:.6f}, 像素Std={luminance_std:.2f}'
                                print(f'  ⚠️ 跳过空图表: 图号{i} (size={file_size} bytes{extra})')

                        except Exception as e:
                            print(f'  ⚠️ 保存图表 {i} 失败: {e}')

                    if kept_images:
                        # 创建因子文件夹
                        factor_dir = os.path.join(self.output_dir, factor_name)
                        os.makedirs(factor_dir, exist_ok=True)
                        
                        for plot_index, img_info in enumerate(kept_images, 1):
                            tmp_path = img_info['tmp_path']
                            final_path = os.path.join(factor_dir, f'{factor_name}_plot_{plot_index}.png')
                            os.replace(tmp_path, final_path)
                            saved_count += 1
                            print(
                                f"  ✓ 已保存: {final_path} ({img_info['file_size']} bytes)"
                                + (
                                    ''
                                    if img_info['non_white_ratio'] is None
                                    else f" | 非白像素占比={img_info['non_white_ratio']:.4f}"
                                )
                            )
                        print(f'✅ 共保存 {saved_count} 个PNG文件')
                    else:
                        print(f'⚠️ 未保存任何PNG文件')
                else:
                    print(f'⚠️  {factor_name} tear-sheet 未生成任何图表')
                
                # 提示：alphalens 可能生成多个图表，第一个可能是空的
                # 调用方需要保存所有图表，不要只保存第一个
                
                # 重要：将保存的图表对象保存到实例变量，供调用方使用
                # 注意：必须保存字典的副本，避免引用被修改
                self._saved_fig_objects = list(captured_figures)
                
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
            
            result = self._quick_score(factor_name, clean, p)
            
            # 检查是否返回了 None（数据不足或计算失败）
            if result is None or result[0] is None:
                print(f'⚠️  {factor_name} 周期{p}天 数据不足，跳过')
                continue
            
            scores, level, ic_series, ret_series, top_turnover = result
            print(f'\n📊 {factor_name} 周期{p}天 打分结果：{level}')
            for k, (v, ok) in scores.items():
                print(f'   {k:12s}: {v:7.3f}  {"✅" if ok else "❌"}')

            # ★ 滚动监控 + 红黄绿灯判定（变量现成）
            latest = self._rolling_monitor(factor_name, ic_series, ret_series, p)
            
            roll_ir   = latest.roll_ir
            win_rate  = (ic_series > 0).rolling(self.cfg.ROLL_WIN if hasattr(self.cfg, 'ROLL_WIN') else 20).mean().iloc[-1]
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


class FactorAnalyzer:
    """因子分析器"""

    def __init__(self, factor_df: pd.DataFrame, start_date: str, end_date: str,
                 stock_pool: str = '000510.XSHG', quantiles: int = 10,
                 periods: List[int] = None, output_dir: str = './results'):
        """
        初始化因子分析器

        Args:
            factor_df: 因子数据 DataFrame，包含 date, asset, factor_value 列
            start_date: 开始日期
            end_date: 结束日期
            stock_pool: 股票池
            quantiles: 分位数
            periods: 调仓周期列表
            output_dir: 输出目录，用于保存图表和结果文件
        """
        self.factor_df = factor_df
        self.start_date = start_date
        self.end_date = end_date
        self.stock_pool = stock_pool
        self.quantiles = quantiles
        self.periods = periods or [5, 10, 15]
        self.output_dir = output_dir

        # 存储分析结果
        self.price_data = None
        self.industry_data = None
        self.clean_data = None
        self.analysis_results = []

        # 创建配置对象供核心方法使用
        self.cfg = self._create_config()
        
        # 创建核心分析器实例
        self.core_analyzer = FactorAnalysisCore(self.cfg, output_dir=self.output_dir)

    def _create_config(self):
        """创建配置对象"""
        class Config:
            def __init__(self, start_date, end_date, stock_pool, quantiles, periods):
                self.START = start_date
                self.END = end_date
                self.STOCK_POOL = stock_pool
                self.QUANTILES = quantiles
                self.PERIODS = periods
                self.MAX_STOCKS = None

        return Config(self.start_date, self.end_date, self.stock_pool, self.quantiles, self.periods)

    def get_price(self) -> pd.DataFrame:
        """
        获取价格数据

        Returns:
            价格数据 DataFrame
        """
        if self.price_data is not None:
            return self.price_data

        # 设置股票列表到核心分析器
        self.core_analyzer.stocks = self._get_stocks_for_calculation()
        
        # 使用核心分析器获取价格数据
        self.price_data = self.core_analyzer.get_price()
        return self.price_data

    def get_industry_series(self) -> pd.Series:
        """
        获取行业分组序列

        Returns:
            行业分组 Series
        """
        if self.industry_data is not None:
            return self.industry_data

        # 设置股票列表到核心分析器
        self.core_analyzer.stocks = self._get_stocks_for_calculation()
        
        # 使用核心分析器获取行业数据
        self.industry_data = self.core_analyzer.get_industry_series()
        return self.industry_data

    def _get_stocks_for_calculation(self) -> List[str]:
        """
        获取用于计算的股票列表

        Returns:
            股票代码列表
        """
        # 从因子数据中获取股票列表
        if self.factor_df is not None and not self.factor_df.empty:
            # 标准化列名以确保能找到正确的股票列
            standardized_df = self._standardize_columns(self.factor_df.copy())
            stocks = standardized_df['asset'].unique().tolist()
            print(f'[配置] 使用因子数据中的 {len(stocks)} 只股票')
            return stocks

        # 如果没有因子数据，从股票池获取
        try:
            if self.stock_pool == 'stock':
                # 获取A股所有股票代码
                stocks = data.get_index_stocks("small")
                print(f'[配置] 获取到A股股票 {len(stocks)} 只')
            else:
                # 获取指数成分股
                symbol = self.stock_pool.split('.')[0]
                stocks = data.get_index_stocks(symbol)
                print(f'[配置] 获取到指数成分股 {len(stocks)} 只')
        except Exception as e:
            print(f'[配置] 获取股票失败: {e}，使用空列表')
            stocks = []

        return stocks

    def analyze_factor(self, factor_name: str = 'analyzed_factor', plot: bool = False) -> List[Dict[str, Any]]:
        """
        分析因子

        Args:
            plot: 是否生成图表

        Returns:
            分析结果列表
        """
        print('正在分析因子...')

        # 准备因子数据
        factor_series = self._prepare_factor_series()

        if factor_series.empty:
            print('⚠️  因子数据为空，无法进行分析')
            return []

        # 创建核心分析器并获取价格和行业数据
        self.core_analyzer.stocks = self._get_stocks_for_calculation()  # 设置股票列表

        price = self.core_analyzer.get_price()
        industry = self.core_analyzer.get_industry_series()

        # 确保因子数据和价格数据的日期范围一致
        if not price.empty and not factor_series.empty:
            # 获取价格数据的日期范围
            price_dates = set(price.index.date)
            
            if price_dates:
                # 过滤因子数据，只保留有价格数据的日期
                date_index = factor_series.index.get_level_values('date')
                mask = pd.Series(date_index).dt.date.isin(price_dates)
                
                # 重新创建 Series，只包含符合条件的行
                filtered_data = {}
                for (date, asset), value in factor_series.items():
                    if pd.Timestamp(date).date() in price_dates:
                        filtered_data[(date, asset)] = value
                factor_series = pd.Series(filtered_data)
                factor_series.index = pd.MultiIndex.from_tuples(factor_series.index, names=['date', 'asset'])
                
                # 过滤行业数据
                industry_filtered_data = {}
                for (date, asset), group in industry.items():
                    if pd.Timestamp(date).date() in price_dates:
                        industry_filtered_data[(date, asset)] = group
                industry = pd.Series(industry_filtered_data, name='group')
                
                print(f'[数据对齐] 共 {len(price_dates)} 个交易日有价格数据，过滤后因子数据: {len(factor_series)} 条')
            else:
                print('⚠️  价格数据为空，无法进行分析')
                return []

        # 使用核心方法处理因子
        factors_dict = {factor_name: factor_series}
        results = self.core_analyzer.process_factors_with_alphalens(factors_dict, price, industry, plot=plot)

        self.analysis_results = results

        print('因子分析完成！')
        return results

    def _prepare_factor_series(self) -> pd.Series:
        """
        准备因子序列数据

        Returns:
            因子序列 (MultiIndex: date, asset)
        """
        if self.factor_df is None or self.factor_df.empty:
            return pd.Series()

        # 复制数据以避免修改原始数据
        df = self.factor_df.copy()

        # 自动检测和标准化列名
        df = self._standardize_columns(df)

        # 转换因子数据为 alphalens 所需的格式
        factor_data = {}
        for _, row in df.iterrows():
            date = pd.Timestamp(row['date'])
            asset = row['asset']
            value = row['factor_value']
            factor_data[(date, asset)] = value

        factor_series = pd.Series(factor_data)
        factor_series.index = pd.MultiIndex.from_tuples(factor_series.index, names=['date', 'asset'])

        return factor_series

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化因子数据列名和格式

        Args:
            df: 原始因子数据框

        Returns:
            标准化后的数据框
        """
        # 自动检测股票代码列
        possible_asset_cols = ['asset', 'code', 'symbol', 'stock_code', 'ticker']
        asset_column = None
        for col in possible_asset_cols:
            if col in df.columns:
                asset_column = col
                break

        if asset_column is None:
            raise ValueError(f"无法找到股票代码列。可用列: {list(df.columns)}")

        # 重命名股票代码列为标准名称
        if asset_column != 'asset':
            df = df.rename(columns={asset_column: 'asset'})

        # 确保股票代码格式正确（6位字符串，前补0）
        df['asset'] = df['asset'].astype(str).str.zfill(6)

        # 验证必要的列是否存在
        required_cols = ['date', 'asset', 'factor_value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要的列: {missing_cols}")

        return df
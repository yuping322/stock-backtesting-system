#!/usr/bin/env python3
"""
板块轮动统计打分与热度追踪 - Python脚本版本
所有A股,剔除ST,停牌,上市不足半年的
针对个股计算指标，N日价格涨幅
group by 板块，给板块热度打分，排序
展示给定日期最热门板块，板块内涨幅靠前个股，绘制板块历史热度情况

风险及免责提示：该策略由聚宽用户在聚宽社区分享，仅供学习交流使用。
原文网址：https://www.joinquant.com/view/community/detail/28005
标题：板块轮动打分与热度追踪
"""

import jq_compat as jq
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import warnings
import time
import matplotlib.pyplot as plt
import sys
import os

warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def time_me(fn):
    def _wrapper(*args, **kwargs):
        start = time.perf_counter()
        ret = fn(*args, **kwargs)
        print("%s cost %.3f second" % (fn.__name__, time.perf_counter() - start))
        return ret
    return _wrapper


@time_me
def stock_industry(date, industry_category_list):
    """
    获得指定日期所有股票的行业分类
    :param date: 日期字符串，如 '2025-11-11'
    :param industry_category_list: ['jq_l1','jq_l2','sw_l1','sw_l2','sw_l3','zjw']
                                  聚宽1级，聚宽2级，申万1级，申万2级，申万3级，证监会行业
    :return: DataFrame 包含股票基本信息和行业分类
    """
    # 过滤新股 (上市不足半年)
    dt = datetime.strptime(date, '%Y-%m-%d')
    df = jq.get_all_securities(types=['stock'], date=date)
    df = df[df['start_date'] < (dt - timedelta(days=180)).date()]

    # 过滤ST股
    try:
        st_df = jq.get_extras('is_st', list(df.index), end_date=date, count=1)
        # st_df的index是股票代码，columns=['is_st']，值为True/False
        non_st_stocks = st_df[st_df['is_st'] == False].index.tolist()
        df_sec = df[df.index.isin(non_st_stocks)]
        print(f"ST过滤后剩余 {len(df_sec)} 只股票")
    except Exception as e:
        print(f"警告: ST数据获取失败，跳过ST过滤: {e}")
        df_sec = df  # 如果ST数据获取失败，使用所有股票

    ind = jq.get_industry(list(df_sec.index), date=date)
    ind_dict = {k: [getattr(ind[k], icode, '其他') for icode in industry_category_list]
                for k in ind}
    df_ind = pd.DataFrame.from_dict(ind_dict, orient='index',
                                    columns=[code for code in industry_category_list])
    df_ret = pd.merge(df_sec, df_ind, left_index=True, right_index=True, how='left')
    return df_ret


def stock_industry_df(date, industry_category_list, df_stock):
    """
    获取给定股票列表，指定日期的行业分类
    :param date: 日期字符串
    :param industry_category_list: 行业分类列表
    :param df_stock: 股票dataframe index是股票代码
    :return: DataFrame
    """
    ind = jq.get_industry(df_stock.index, date=date)
    ind_dict = {k: [ind[k][industry_category]['industry_name'] for industry_category in industry_category_list if
                    len(set(ind[k].keys()).intersection(set(industry_category_list))) == len(industry_category_list)]
                for k in ind}
    df_ind = pd.DataFrame.from_dict(ind_dict, orient='index',
                                    columns=[code for code in industry_category_list])
    df = pd.merge(df_stock, df_ind, left_index=True, right_index=True, how='left')
    return df


@time_me
def returns(stock_df, date, days):
    """
    计算N日收益率
    :param stock_df: 股票DataFrame
    :param date: 日期字符串，如 '2025-11-11'
    :param days: 计算天数，如 5
    :return: 包含收益率的DataFrame
    """
    panel = jq.get_price(list(stock_df.index), end_date=date, frequency='daily',
                        fields=['close'], adjust='qfq', count=days+1)

    if isinstance(panel, dict):
        # 如果返回字典，合并所有股票的数据
        if panel:
            df_close = pd.DataFrame()
            for stock_code, stock_data in panel.items():
                if not stock_data.empty:
                    temp_df = stock_data[['close']].copy()
                    temp_df.columns = [stock_code]
                    if df_close.empty:
                        df_close = temp_df
                    else:
                        df_close = df_close.join(temp_df, how='outer')
        else:
            df_close = pd.DataFrame()
    else:
        # 如果返回单个DataFrame
        df_close = panel[['close']] if not panel.empty else pd.DataFrame()

    if not df_close.empty and len(df_close) > days:
        # 计算收益率：(最新价 / days天前的价格) - 1
        series_return = (df_close.iloc[-1] / df_close.iloc[-days-1]) - 1
        stock_df['return'] = series_return

    return stock_df


@time_me
def returns_series(stock_df, date, days, back_days):
    """
    计算连续back_days天的days日收益率
    :param stock_df: 股票DataFrame
    :param date: 日期字符串
    :param days: 收益率计算天数
    :param back_days: 回溯天数
    :return: 包含多日收益率的DataFrame
    """
    panel = jq.get_price(list(stock_df.index), end_date=date, frequency='daily',
                        fields=['close'], adjust='qfq', count=days+back_days+1)

    if panel is None:
        # 如果get_price返回None，返回原始DataFrame
        return stock_df

    if isinstance(panel, dict):
        # 如果返回字典，合并所有股票的数据
        if panel:
            df_close = pd.DataFrame()
            for stock_code, stock_data in panel.items():
                if stock_data is not None and not stock_data.empty:
                    temp_df = stock_data[['close']].copy()
                    temp_df.columns = [stock_code]
                    if df_close.empty:
                        df_close = temp_df
                    else:
                        df_close = df_close.join(temp_df, how='outer')
        else:
            df_close = pd.DataFrame()
    else:
        # 如果返回单个DataFrame
        df_close = panel[['close']] if panel is not None and not panel.empty else pd.DataFrame()

    if not df_close.empty and len(df_close) >= (days + back_days):
        series_returns = []
        for i in range(0, back_days):
            # 计算每一天的days日收益率
            ret = (df_close.iloc[-(back_days+1-i)] / df_close.iloc[-(back_days+1-i+days)]) - 1
            series_returns.append(ret)
            stock_df['return_' + str(i+1)] = ret

    return stock_df


def group_score(df, change_limit, back_days):
    """
    板块得分根据最后一天计算的收益率得到
    :param df: 板块内股票DataFrame
    :param change_limit: 涨幅阈值(%)
    :param back_days: 回溯天数
    :return: 板块得分
    """
    return_col = 'return_' + str(back_days)
    if return_col not in df.columns:
        return 0

    df_ok = df[df[return_col] > change_limit/100]
    # 超过收益率阈值的股票上涨均值
    mean = df_ok[return_col].mean() * 100 if len(df_ok) > 0 and len(df) > 10 else 0
    # 行业内超过收益率阈值股票的平均收益率*超过阈值股票个数占行业比例
    score = int(len(df_ok) / len(df) * mean * 10)
    return score


def group_score_series(df, change_limit, days, back_days):
    """
    板块热度得分序列
    :param df: 板块内股票DataFrame
    :param change_limit: 涨幅阈值(%)
    :param days: 收益率计算天数
    :param back_days: 回溯天数
    :return: 得分序列列表
    """
    scores = []
    for i in range(0, back_days):
        return_col = 'return_' + str(i+1)
        if return_col not in df.columns:
            scores.append(0)
            continue

        df_ok = df[df[return_col] > change_limit/100]
        mean = df_ok[return_col].mean() * 100 if len(df_ok) > 0 and len(df) > 10 else 0
        score = int(len(df_ok) / len(df) * mean * 10)
        scores.append(score)
    return scores


def group_top_list(df, top_count, change_limit, days):
    """
    板块days日收益率排名前top_count的股票
    :param df: 板块内股票DataFrame
    :param top_count: 返回前N名股票
    :param change_limit: 涨幅阈值(%)
    :param days: 收益率计算天数
    :return: 格式化的字符串
    """
    return_col = 'return_' + str(days)
    if return_col not in df.columns:
        return "[]"

    df_top = df.sort_values(by=return_col, ascending=False).head(top_count)
    df_top['print'] = df_top['display_name'] + ":" + (df_top[return_col] * 100).round(2).astype('str') + "%"
    df_ok = df[df[return_col] > change_limit/100]
    return "[" + str(len(df_ok)) + "/" + str(len(df)) + "]" + ",".join(list(df_top['print']))


@time_me
def top_industry(date, industry_category='sw_l2', return_days=5, back_days=10,
                stock_top_count=5, industry_top_count=10, up_limit=10):
    """
    计算当前根据return_days收益率计算的行业热度指数前industry_top_count名,
    并展示这些行业最近back_days日的得分
    :param date: 日期字符串，如 '2025-11-11'
    :param industry_category: 行业分类，如 'sw_l1', 'sw_l2', 'sw_l3', 'zjw', 'jq_l1', 'jq_l2'
    :param return_days: 收益率计算天数，默认5
    :param back_days: 回溯天数，默认10
    :param stock_top_count: 展示前多少位股票，默认5
    :param industry_top_count: 展示前多少行业，默认10
    :param up_limit: 涨幅限制(%)，默认10
    :return: None (直接打印结果)
    """
    print(f"\n=== 板块轮动分析 ===")
    print(f"日期: {date}")
    print(f"行业分类: {industry_category}")
    print(f"收益率计算天数: {return_days}")
    print(f"回溯天数: {back_days}")
    print(f"涨幅阈值: {up_limit}%")
    print("=" * 50)

    df_ind = stock_industry(date, [industry_category])
    print(f"获取到 {len(df_ind)} 只股票")

    if df_ind.empty:
        print("警告: 未获取到任何股票数据，请检查数据源或日期")
        return

    df_return = returns_series(df_ind, date, return_days, back_days)
    print(f"计算完成收益率数据")

    # 如果没有有效的收益率数据，创建模拟数据用于测试
    if df_return.empty or not any(col.startswith('return_') for col in df_return.columns):
        print("警告: 没有有效的收益率数据，使用模拟数据进行测试")
        # 创建模拟数据
        np.random.seed(42)  # 固定随机种子以获得可重复的结果

        # 为每个板块创建模拟的收益率数据
        unique_industries = df_ind[industry_category].unique()
        for industry in unique_industries:
            industry_mask = df_ind[industry_category] == industry
            industry_stocks = df_ind[industry_mask]

            if len(industry_stocks) > 0:
                # 为这个板块的股票生成模拟收益率
                for i in range(1, back_days + 1):
                    col_name = f'return_{i}'
                    # 生成-10%到+10%之间的随机收益率
                    simulated_returns = np.random.normal(0.005, 0.03, len(industry_stocks))
                    df_return.loc[industry_mask, col_name] = simulated_returns

        print("模拟数据创建完成")

    s_tops = df_return.groupby(industry_category).apply(group_top_list, stock_top_count, up_limit, return_days)
    s_scores = df_return.groupby(industry_category).apply(group_score_series, up_limit, return_days, back_days)
    s_score = df_return.groupby(industry_category).apply(group_score, up_limit, back_days)

    if s_tops.empty:
        print("警告: 没有有效的板块数据")
        return

    df = pd.DataFrame({"scores": s_scores, "top": s_tops, "score": s_score},
                     index=s_tops.index).sort_values(by='score', ascending=False).head(industry_top_count)

    print(f"\n前 {industry_top_count} 热门板块:")
    print("-" * 80)
    for idx, row in df.iterrows():
        print(f"板块: {idx}")
        print(f"热度得分: {row['score']}")
        print(f"股票详情: {row['top']}")
        print("-" * 80)

    # 绘制热度图
    plot_industry_hot(df, back_days, date, industry_category, return_days, back_days)


def plot_industry_hot(df, days, date, industry_category='sw_l2', return_days=5, back_days=10):
    """
    绘制板块热度指数图表
    :param df: 包含板块得分的DataFrame
    :param days: 回溯天数
    :param date: 日期字符串
    :param industry_category: 行业分类
    :param return_days: 收益率天数
    :param back_days: 回溯天数
    """
    plt.figure(figsize=(15, 10))
    x = [i for i in range(1, days+1)]
    plt.title(f'板块热度指数: {date}', fontsize=16, fontweight='bold')

    # 定义颜色和标记
    colors = plt.cm.tab10.colors
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd']

    for i, row in enumerate(df.itertuples()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        plt.plot(x, row.scores, label=row.Index, color=color,
                marker=marker, markersize=6, linewidth=2, alpha=0.8)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('天数', fontsize=12)
    plt.ylabel('板块热度得分', fontsize=12)
    plt.xticks(x)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图片
    plt.savefig(f'sector_hot_{date}_{industry_category}_{return_days}d_{back_days}back.png',
                dpi=300, bbox_inches='tight')
    print(f"\n图表已保存为: sector_hot_{date}_{industry_category}_{return_days}d_{back_days}back.png")

    plt.show()


def main():
    """主函数 - 运行板块轮动分析"""
    # 默认参数
    default_date = '2025-11-11'
    default_category = 'sw_l2'
    default_return_days = 5
    default_back_days = 30
    default_up_limit = 30

    # 可以通过命令行参数覆盖默认值
    if len(sys.argv) > 1:
        if len(sys.argv) != 6:
            print("使用方法: python sector_rotation_analysis.py [日期] [行业分类] [收益率天数] [回溯天数] [涨幅阈值]")
            print("示例: python sector_rotation_analysis.py 2025-11-11 sw_l2 5 30 30")
            print("\n行业分类选项: sw_l1, sw_l2, sw_l3, zjw, jq_l1, jq_l2")
            sys.exit(1)

        date = sys.argv[1]
        industry_category = sys.argv[2]
        return_days = int(sys.argv[3])
        back_days = int(sys.argv[4])
        up_limit = float(sys.argv[5])
    else:
        print("使用默认参数运行...")
        date = default_date
        industry_category = default_category
        return_days = default_return_days
        back_days = default_back_days
        up_limit = default_up_limit

    # 运行分析
    top_industry(date, industry_category, return_days, back_days, 5, 10, up_limit)


if __name__ == "__main__":
    main()
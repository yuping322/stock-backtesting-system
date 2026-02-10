# 📊 因子检验结果总结

## 测试配置
- **测试期间**: 2025-07-26 ~ 2025-10-24 (3个月)
- **股票池**: small (小盘股)
- **测试因子数**: 252个
- **有效因子数**: 约60个有效因子

## 🎯 快速选股组合

### 顶级有效因子 (7分以上)

| 因子 | 类型 | 最佳周期 | 得分 | 说明 |
|------|------|---------|------|------|
| **sales_growth** | 📈 增长 | 5/10/15天 | 8/9 | 销售增长率 - 最强 |
| operating_cost_ttm | 💰 盈利 | 10天 | 7/9 | 营业成本 |
| total_operating_revenue_ttm | 💰 盈利 | 10天 | 7/9 | 营业收入 |
| gross_profit_ttm | 💰 盈利 | 10天 | 7/9 | 毛利润 |
| boll_down | 📊 技术 | 15天 | 7/9 | 布林下轨 |
| MAC20 | 📊 技术 | 15天 | 7/9 | 移动平均 |
| market_cap | 📏 规模 | 5天 | 7/9 | 市值 |
| raw_beta | 📊 技术 | 5天 | 7/9 | 市场Beta |
| beta | 📊 技术 | 5天 | 7/9 | 市场Beta |
| size | 📏 规模 | 5天 | 7/9 | 规模 |

### 次优有效因子 (6分)

| 因子 | 类型 | 最佳周期 |
|------|------|---------|
| total_profit_growth_rate | 📈 增长 | 10天 |
| np_parent_company_owners_growth_rate | 📈 增长 | 5天 |
| total_profit_ttm | 💰 盈利 | 15天 |
| np_parent_company_owners_ttm | 💰 盈利 | 5天 |
| net_profit_ttm | 💰 盈利 | 15天 |
| goods_sale_and_service_render_cash_ttm | 💵 现金流 | 10天 |
| operating_revenue_ttm | 💰 盈利 | 10天 |
| financial_liability | 📋 负债 | 10天 |
| interest_free_current_liability | 📋 负债 | 10天 |
| total_asset_growth_rate | 📈 增长 | 5天 |
| operating_revenue_growth_rate | 📈 增长 | 15天 |
| EBITDA | 💰 盈利 | 5天 |
| gross_profit_ttm | 💰 盈利 | 10天 |
| EBIT | 💰 盈利 | 5天 |
| operating_liability | 📋 负债 | 15天 |
| administration_expense_ttm | 💰 盈利 | 10天 |
| MAC10 | 📊 技术 | 10天 |
| SGI | 📈 增长 | 5天 |

## 有效因子分类总结

### 🟢 财务质量类因子 (Profitability & Earnings Quality)

| 因子名称 | 最佳周期 | 通过指标 | 说明 |
|---------|---------|---------|------|
| operating_cost_ttm | 10天 | 7/9 | 营业成本TTM - 成本控制能力 |
| total_operating_revenue_ttm | 10天 | 7/9 | 营业收入TTM - 收入规模 |
| gross_profit_ttm | 10天 | 7/9 | 毛利润TTM - 盈利能力 |
| total_profit_ttm | 15天 | 7/9 | 利润总额TTM - 综合盈利 |
| net_profit_ttm | 15天 | 7/9 | 净利润TTM - 最终盈利 |
| EBITDA | 5天 | 5/9 | 息税折旧前利润 - 现金流质量 |
| EBIT | 5天 | 5/9 | 息税前利润 |
| eps_ttm | 15天 | 5/9 | 每股收益TTM - 盈利质量 |
| operating_profit_per_share | 5天 | 4/9 | 每股营业利润 |
| retained_earnings | 5天 | 5/9 | 留存收益 - 积累能力 |
| retained_earnings_per_share | 5天 | 4/9 | 每股留存收益 |
| non_recurring_gain_loss | 5天 | 5/9 | 非经常性损益 - 盈利能力质量 |
| OperateNetIncome | 5天 | 4/9 | 营业净利润 |

### 🟢 现金流类因子 (Cash Flow Quality)

| 因子名称 | 最佳周期 | 通过指标 | 说明 |
|---------|---------|---------|------|
| cashflow_per_share_ttm | 5天 | 5/9 | 每股现金流TTM - 现金流质量 |
| cash_flow_to_price_ratio | 5天 | 5/9 | 现金流市价比 - 性价比 |
| net_operate_cash_flow_per_share | 15天 | 5/9 | 每股经营现金流 - 经营质量 |
| cash_and_equivalents_per_share | 5天 | 4/9 | 每股现金等价物 |
| goods_sale_and_service_render_cash_ttm | 10天 | 6/9 | 销售商品服务收到现金TTM |

### 🟢 规模与估值类因子 (Size & Value)

| 因子名称 | 最佳周期 | 通过指标 | 说明 |
|---------|---------|---------|------|
| market_cap | 5天 | 7/9 | 市值 - 规模因子 |
| circulating_market_cap | 5天 | 7/9 | 流通市值 |
| size | 5天 | 7/9 | 规模因子 |
| raw_beta | 5天 | 7/9 | 原始Beta - 系统性风险 |
| beta | 5天 | 7/9 | 市场Beta |
| natural_log_of_market_cap | 5天 | 7/9 | 市值对数 |
| MAC20 | 15天 | 7/9 | 20日移动平均成本 |

### 🟢 增长类因子 (Growth)

| 因子名称 | 最佳周期 | 通过指标 | 说明 |
|---------|---------|---------|------|
| operating_revenue_growth_rate | 15天 | 7/9 | 营收增长率 - 成长性 |
| total_profit_growth_rate | 10天 | 6/9 | 利润增长率 |
| np_parent_company_owners_growth_rate | 5天 | 5/9 | 归母净利润增长率 |
| total_asset_growth_rate | 5天 | 6/9 | 总资产增长率 |
| operating_profit_growth_rate | 10天 | 6/9 | 营业利润增长率 |
| sales_growth | 5天 | 8/9 | 销售增长率 - 最强成长因子 |
| growth | 5天 | 6/9 | 综合增长率 |
| SGI | 5天 | 6/9 | 销售增长指标 |

### 🟢 资产负债质量类因子 (Balance Sheet Quality)

| 因子名称 | 最佳周期 | 通过指标 | 说明 |
|---------|---------|---------|------|
| financial_liability | 10天 | 6/9 | 金融负债 |
| interest_free_current_liability | 10天 | 6/9 | 无息流动负债 |
| interest_carry_current_liability | 5天 | 5/9 | 有息流动负债 |
| operating_liability | 15天 | 7/9 | 经营性负债 |
| net_working_capital | 5天 | 4/9 | 净营运资本 |
| LVGI | 5天 | 5/9 | 杠杆率 |

### 🟢 每股指标类因子 (Per Share Metrics)

| 因子名称 | 最佳周期 | 通过指标 | 说明 |
|---------|---------|---------|------|
| total_operating_revenue_per_share_ttm | 5天 | 4/9 | 每股营业收入TTM |
| capital_reserve_fund_per_share | 5天 | 4/9 | 每股资本公积 |
| net_asset_per_share | 5天 | 4/9 | 每股净资产 |
| operating_profit_per_share_ttm | 15天 | 5/9 | 每股营业利润TTM |
| total_operating_revenue_per_share | 15天 | 5/9 | 每股营业收入 |
| total_operating_cost_ttm | 10天 | 6/9 | 营业成本TTM |

### 🟢 资产效率类因子 (Asset Efficiency)

| 因子名称 | 最佳周期 | 通过指标 | 说明 |
|---------|---------|---------|------|
| total_asset_turnover_rate | 10天 | 5/9 | 总资产周转率 |
| current_asset_turnover_rate | 10天 | 4/9 | 流动资产周转率 |
| asset_impairment_loss_ttm | 15天 | 6/9 | 资产减值损失TTM |

### 🟢 技术指标类因子 (Technical Indicators)

| 因子名称 | 最佳周期 | 通过指标 | 说明 |
|---------|---------|---------|------|
| boll_down | 15天 | 7/9 | 布林下轨 - 均值回归 |
| MAC20 | 15天 | 7/9 | 移动平均成本 |
| EMAC10 | 10天 | 5/9 | 指数移动平均成本 |
| EMAC12 | 10天 | 6/9 | 指数移动平均成本12 |

### 🟢 其他有效因子

| 因子名称 | 最佳周期 | 通过指标 | 说明 |
|---------|---------|---------|------|
| net_interest_expense | 5天 | 5/9 | 净利息费用 |
| administration_expense_ttm | 10天 | 6/9 | 管理费用TTM |
| np_parent_company_owners_ttm | 5天 | 6/9 | 归母净利润TTM |
| price_no_fq | 5天 | 5/9 | 不复权价格 |

## 统计汇总

### 按因子类型统计

| 类型 | 因子数 | 占比 |
|------|--------|------|
| 财务质量类 | 13 | 33.3% |
| 现金流类 | 5 | 12.8% |
| 规模与估值类 | 7 | 17.9% |
| 增长类 | 8 | 20.5% |
| 资产负债质量类 | 6 | 15.4% |
| 每股指标类 | 6 | 15.4% |
| 资产效率类 | 3 | 7.7% |
| 技术指标类 | 4 | 10.3% |

### 按最佳周期分布

| 周期 | 因子数 |
|------|--------|
| 5天 | 25个 |
| 10天 | 30个 |
| 15天 | 20个 |

## 推荐组合

### 高评分因子推荐（7/9以上）

1. **sales_growth** (8/9) - 销售增长率，最强因子
2. **operating_cost_ttm** (7/9) - 营业成本
3. **total_operating_revenue_ttm** (7/9) - 营业收入
4. **gross_profit_ttm** (7/9) - 毛利润
5. **boll_down** (7/9) - 布林下轨技术指标
6. **MAC20** (7/9) - 移动平均成本
7. **market_cap** (7/9) - 市值规模
8. **circulating_market_cap** (7/9) - 流通市值
9. **size** (7/9) - 规模因子
10. **raw_beta** (7/9) - 市场Beta
11. **beta** (7/9) - 市场Beta
12. **natural_log_of_market_cap** (7/9) - 市值对数

### 推荐组合策略

#### 基本面组合（3个月验证有效）
- **growth** (增长) + **profitability** (盈利) + **cash flow** (现金流) + **size** (规模)

具体因子：
- sales_growth (销售增长)
- operating_revenue_growth_rate (营收增长)
- eps_ttm (每股收益)
- net_profit_ttm (净利润)
- cashflow_per_share_ttm (每股现金流)
- market_cap (市值)

#### 技术面组合
- **boll_down** (布林下轨) + **MAC20** (移动平均) + **beta** (Beta)

## 注意事项

1. **3个月数据**: 这些因子在过去3个月有效，但需要持续监控
2. **周期性**: 不同周期(5/10/15天)因子表现不同，需灵活调整
3. **组合使用**: 建议多因子组合使用，降低单一因子失效风险
4. **持续监控**: 定期检查因子有效性，防止因子衰减

## 下一步建议

1. **用6-12个月历史数据训练**: 选中基本面组合的因子
2. **生成预测**: 使用 `scripts/build_predictions.py` 生成预测
3. **回测验证**: 用 `BacktestEngine` 验证策略效果
4. **定期更新**: 每月重新运行因子检验，更新因子池


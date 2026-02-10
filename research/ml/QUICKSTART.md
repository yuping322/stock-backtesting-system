# 快速开始

## 1. 训练模型（首次运行）

```bash
# 使用历史数据训练超级因子模型
python ml/train_super_factor.py
```

等待完成，模型会保存在 `ml/models/` 目录。

## 2. 预测选股（每日运行）

```bash
# 自动选择最新交易日的数据
python ml/predict_stocks.py --top-n 10

# 指定日期
python ml/predict_stocks.py --date 2024-10-24 --top-n 10
```

结果保存到 `data/factor_values_sample.csv`

## 3. 查看结果

```bash
# 查看最新的选股结果
tail -20 data/factor_values_sample.csv
```

## 示例输出

```
date,code,weight
2024-10-24,000001,0.0919
2024-10-24,000002,0.0649
2024-10-24,000003,0.0776
2024-10-24,000004,0.0966
2024-10-24,000005,0.0610
...
```

## 定时任务设置

### 每日收盘后自动运行

创建 `scripts/daily_predict.sh`:

```bash
#!/bin/bash
cd /path/to/stock-backtesting-system
python ml/predict_stocks.py --top-n 10 >> logs/daily_predict.log 2>&1
```

添加到 crontab:
```bash
crontab -e
# 添加这一行：每天16:00运行
0 16 * * * /path/to/scripts/daily_predict.sh
```

## 常见问题

### Q: 训练需要多长时间？
A: 根据数据量，通常5-15分钟。

### Q: 预测需要多长时间？
A: 通常1-2分钟。

### Q: 多久重新训练一次？
A: 建议每月或每季度重训一次。

### Q: 如何修改选股数量？
A: 使用 `--top-n` 参数，例如 `--top-n 20`。

### Q: 模型文件在哪里？
A: `ml/models/` 目录。

## 下一步

1. ✅ 训练模型
2. ✅ 每日预测
3. 📊 评估效果（回测）
4. 🔄 定期更新模型

详细文档请查看 `ml/README.md`


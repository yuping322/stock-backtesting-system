# 技术设计：数据服务 (DataMCP + Skills) - 极简双层架构版

响应“简单、直接、高效”的设计理念，我们将数据服务架构精简为 **双层 (Two-Layer)** 结构。

## 1. 核心架构：双层模型

### Layer 1: 本地数据仓库 (Local Data Warehouse - DuckDB)
这是系统的**核心**。所有上层业务（回测、因子计算）**只**与这一层交互。
- **存储介质**: DuckDB 文件。
- **分库分表策略**:
    - **`market.db` (行情库)**:
        - `stock_daily`: 日线行情 (OHLCV)。
        - `index_daily`: 指数行情。
        - *特点*: 数据量大，读写频繁，按日追加。
    
    - **`finance.db` (财务库 - 对应 AkShare/Sina)**:
        - `indicator_quarterly`: 财务指标宽表。
        - *Schema*: `code, report_date, publish_date, eps, roe, net_assets_ps, ...`
        - *特点*: 字段多但行数少（每季一条），适合宽表存储。

    - **`industry.db` (经营数据库 - 对应 Lixinger 非财务)**:
        - `business_metrics`: 行业经营数据 (EAV模型)。
        - *Schema*: `code, date, metric_name (e.g., '门店数'), value, unit`。
        - *特点*: 字段极度稀疏（不同行业指标完全不同），适合长表存储 (Entity-Attribute-Value)。

    - **`meta.db` (元数据库)**:
        - `stock_info`: 股票基础信息。
        - `calendar`: 交易日历。

- **存储优化**:
    - 均采用 DuckDB 原生 `.db` 文件格式。
    - 针对 `market.db` 的 `date` 列建立索引，加速区间查询。
- **按需加载**: DuckDB 支持 `ATTACH` 命令，可以在查询时动态挂载需要的数据库文件，无需一次性加载全量数据。

### Layer 2: 远程数据接入层 (Remote Data Connectors)
这是数据的**来源**。只负责“抓取 -> 清洗 -> 写入 DuckDB”。
- **Connectors**:
    - `OSSFetcher`: 阿里云 OSS 下载器。
    - `LixingrenFetcher`: 理杏仁数据接口。
    - `AkShareFetcher`: AkShare 接口封装。
    - `ExternalMCPFetcher`: **[新增]** 对接第三方现成的 MCP 数据服务 (作为 Client 消费其 Tools)。
- **工作模式**:
    - **并集写入**: 多个源的数据经过标准化后，统一 `UPSERT` 到 DuckDB。
    - **定时任务**: 每天收盘后触发一次全量增量同步。

### *关于内存缓存*
- **原则**: 数据服务层 **不维护** 复杂的内存缓存。
- **实现**: 业务层（如回测引擎）如果有极高频的重复访问需求，应在业务层自己维护 `lru_cache` 或 `DataFrame` 缓存。数据服务的职责是**极速地从 DuckDB 返回数据**。

---

## 2. 数据流向 (Data Flow)

### 场景 A：日常回测 (Read)
1. 策略 MCP 请求 `get_ohlcv(code='000001', start='2023-01-01')`。
2. 数据服务连接 `market_data.db`。
3. 执行 SQL: `SELECT * FROM stock_daily WHERE code='000001' AND date >= ...`。
4. 返回 DataFrame。
*(全过程不涉及网络请求，保证回测速度)*

### 场景 B：盘后同步 (Write)
1. 定时任务触发 `sync_daily_data()`。
2. `LixingrenFetcher` 抓取今日财务公告 -> 写入 `financial_data.db`。
3. `OSSFetcher` 抓取今日行情 -> 写入 `market_data.db`。
4. 完成。

---

## 3. MCP 接口 (Tools)

保持高层接口稳定，底层实现切换为直连 DuckDB。

### 核心读接口
- **`get_ohlcv_data`**:SQL 查询 `market_data.db`。
- **`get_financial_report`**: SQL 查询 `financial_data.db`。
- **`get_stock_list`**: SQL 查询 `meta_data.db`。

### 核心写接口
- **`sync_remote_data(sources=['oss', 'lixingren'])`**: 触发指定源的同步流程。
- **`update_local_db(table_name, dataframe)`**: 手动向 DuckDB 灌入数据（用于临时修复或导入私有数据）。

### 管理接口
- **`inspect_db_status(db_name)`**: 查看指定数据库文件的大小、表结构、最新日期。
- **`vacuum_db(db_name)`**: 执行 `VACUUM` 命令，整理 DuckDB 碎片，优化文件大小。

---

## 4. 目录结构 (src/data 重构)

```text
src/data/
├── __init__.py
├── api.py                # 暴露给 MCP 的统一读写接口
├── warehouse/            # 本地仓库层
│   ├── connector.py      # DuckDB 连接池管理
│   └── schema.py         # 建表语句
└── providers/            # 远程接入层
    ├── oss.py
    ├── lixingren.py
    ├── akshare.py
    └── mcp_client.py     # 第三方 MCP 调用客户端
```

这种架构的优势：
1. **简单**: 代码逻辑只有“读库”和“写库”两件事。
2. **解耦**: 换数据源只需写一个新的 Provider，不影响读接口。
3. **高效**: DuckDB 的列式存储对 OLAP 查询（如拉取某股票3年行情）本身就极快，甚至不需要额外的内存缓存。

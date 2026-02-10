import akshare as ak
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
import hashlib
import fcntl
import time
import os
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 延迟导入可选依赖
def _get_oss2():
    """延迟导入 oss2"""
    if 'oss2' not in sys.modules:
        try:
            import oss2
        except ImportError:
            return None
    return sys.modules.get('oss2')

# 导入优化组件
try:
    from config_manager import get_config_manager
    from parameter_filter import get_parameter_filter
    _has_optimization = True
except ImportError:
    _has_optimization = False
    logger.warning("Optimization components not available, using fallback mode")

class AkShareWrapperV2:
    def _acquire_lock(self, file_path: Path, timeout: int = 10) -> Optional[int]:
        """
        获取文件独占锁
        
        Args:
            file_path: 文件路径
            timeout: 超时时间（秒）
            
        Returns:
            文件描述符，如果获取失败返回None
        """
        try:
            # 确保父目录存在
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 以读写模式打开文件，如果不存在会创建
            fd = os.open(file_path, os.O_RDWR | os.O_CREAT, 0o644)
            
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return fd
                except BlockingIOError:
                    time.sleep(0.1)  # 等待100ms后重试
            logger.warning(f"Failed to acquire lock for {file_path} within {timeout} seconds")
            os.close(fd)
            return None
        except Exception as e:
            logger.error(f"Error acquiring lock for {file_path}: {e}")
            return None
    
    def _release_lock(self, fd: int):
        """
        释放文件锁
        
        Args:
            fd: 文件描述符
        """
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        except Exception as e:
            logger.error(f"Error releasing lock: {e}")
    
    def _write_meta_with_lock(self, meta_path: Path, data: dict, timeout: int = 5) -> bool:
        """
        使用文件锁安全地写入元数据文件
        
        注意：此方法假设调用者已经持有锁。如果需要获取锁，请使用 _read_write_meta_with_lock
        
        Args:
            meta_path: 元数据文件路径
            data: 要写入的数据
            timeout: 锁超时时间（秒）- 此参数已废弃，因为假设已持有锁
            
        Returns:
            bool: 是否写入成功
        """
        storage_type = self.storage_config.get('type', 'local')
        
        if storage_type == 'local':
            # 本地存储模式：只写入本地文件
            try:
                # 确保父目录存在
                meta_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 使用临时文件写入，然后原子性替换
                temp_path = meta_path.with_suffix('.tmp')
                
                # 先写入临时文件
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    # 确保数据写入磁盘
                    f.flush()
                    os.fsync(f.fileno())
                
                # 原子性替换原文件
                temp_path.replace(meta_path)
                
                # 确保目录操作同步到磁盘
                dir_fd = os.open(str(meta_path.parent), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
                
                return True
                
            except Exception as e:
                logger.error(f"Error writing meta file {meta_path}: {e}")
                # 清理临时文件
                if temp_path.exists():
                    temp_path.unlink()
                return False
                
        elif storage_type == 'oss':
            # OSS存储模式：只上传到OSS，不写入本地文件
            try:
                # 创建临时文件用于上传
                temp_path = meta_path.with_suffix('.tmp')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                # 上传到OSS
                oss_key = self._get_oss_key(str(meta_path.relative_to(self.base_path)))
                success = self._upload_to_oss(temp_path, oss_key)
                
                # 清理临时文件
                if temp_path.exists():
                    temp_path.unlink()
                
                if success:
                    logger.info(f"Successfully uploaded meta file to OSS: {oss_key}")
                    return True
                else:
                    logger.error(f"Failed to upload meta file to OSS: {oss_key}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error uploading meta file to OSS {meta_path}: {e}")
                # 清理临时文件
                if temp_path.exists():
                    temp_path.unlink()
                return False
        else:
            logger.error(f"Unsupported storage type: {storage_type}")
            return False
    def _normalize_dataframe_for_parquet(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        规范化DataFrame以确保兼容Parquet格式
        
        Args:
            df: 输入的DataFrame
            
        Returns:
            规范化后的DataFrame
        """
        if df.empty:
            return df
            
        # 创建副本避免修改原数据
        df_normalized = df.copy()
        
        # 处理混合类型列（特别是object类型列包含多种数据类型的情况）
        for col in df_normalized.columns:
            if df_normalized[col].dtype == 'object':
                # 检查该列是否包含混合类型
                unique_types = set(type(x).__name__ for x in df_normalized[col].dropna())
                if len(unique_types) > 1:
                    # 如果包含多种类型，统一转换为字符串
                    df_normalized[col] = df_normalized[col].astype(str)
                elif unique_types == {'int'}:
                    # 如果都是整数，保持为int64
                    try:
                        df_normalized[col] = df_normalized[col].astype('int64')
                    except (ValueError, TypeError):
                        # 如果转换失败，转为字符串
                        df_normalized[col] = df_normalized[col].astype(str)
                elif unique_types == {'float'}:
                    # 如果都是浮点数，保持为float64
                    try:
                        df_normalized[col] = df_normalized[col].astype('float64')
                    except (ValueError, TypeError):
                        # 如果转换失败，转为字符串
                        df_normalized[col] = df_normalized[col].astype(str)
                elif unique_types == {'str'}:
                    # 如果都是字符串，保持为字符串
                    df_normalized[col] = df_normalized[col].astype(str)
                # 其他情况保持不变
        
        return df_normalized

    def _meta_path_regular(self, interface: str, symbol: str, params_key: str) -> Path:
        """Regular (non-time-series) interface metadata path"""
        fname = f"{interface}_{symbol}_{params_key}.meta.json"
        return self.base_path / interface / symbol / fname

    def _data_path_regular(self, interface: str, symbol: str, params_key: str) -> Path:
        """Regular (non-time-series) interface data path"""
        fname = f"{interface}_{symbol}_{params_key}.parquet"
        return self.base_path / interface / symbol / fname

    def _load_regular_metadata(self, interface: str, symbol: str, params_key: str) -> Optional[dict]:
        meta_path = self._meta_path_regular(interface, symbol, params_key)
        
        storage_type = self.storage_config.get('type', 'local')
        
        if storage_type == 'local':
            # 本地存储模式：只从本地读取
            if meta_path.exists():
                # 使用文件锁确保并发安全
                fd = self._acquire_lock(meta_path, timeout=5)
                if fd is None:
                    logger.warning(f"Failed to acquire lock for reading meta file: {meta_path}")
                    return None
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Error loading meta file {meta_path}: {e}")
                    return None
                finally:
                    self._release_lock(fd)
            return None
            
        elif storage_type == 'oss':
            # OSS存储模式：只从OSS下载
            oss_key = self._get_oss_key(str(meta_path.relative_to(self.base_path)))
            if self._exists_in_oss(oss_key):
                if self._download_from_oss(oss_key, meta_path):
                    logger.debug(f"Downloaded meta file from OSS: {oss_key}")
                    # 下载后读取
                    try:
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            return json.load(f)
                    except Exception as e:
                        logger.error(f"Error loading downloaded meta file {meta_path}: {e}")
                        return None
                else:
                    logger.warning(f"Failed to download meta file from OSS: {oss_key}")
            return None
        else:
            logger.error(f"Unsupported storage type: {storage_type}")
            return None

    def _should_refresh_configured(self, interface: str, symbol: str, params_key: str) -> Tuple[bool, bool]:
        """
        检查是否应该根据配置刷新缓存
        
        Returns:
            Tuple[bool, bool]: (是否需要刷新, 是否有配置)
        """
        if not self.config_manager:
            return False, False
        
        try:
            # 从配置管理器获取刷新配置
            refresh_config = self.config_manager.get_refresh_config(interface, symbol)
            if refresh_config:
                # 检查是否在刷新间隔内
                last_refresh = refresh_config.get('last_refresh')
                interval_minutes = refresh_config.get('interval_minutes', 60)
                
                if last_refresh:
                    last_refresh_dt = datetime.fromisoformat(last_refresh)
                    time_diff = (datetime.now() - last_refresh_dt).total_seconds() / 60
                    return time_diff >= interval_minutes, True
            
            return False, False
        except Exception as e:
            logger.warning(f"Error checking configured refresh for {interface}:{symbol}: {e}")
            return False, False

    def _handle_regular(self, interface: str, symbol: str, params_key: str, kwargs: dict) -> pd.DataFrame:
        """非时间序列接口自适应缓存主流程"""
        # 跟踪使用频率
        self._track_usage(interface, symbol, params_key)
        
        meta = self._load_regular_metadata(interface, symbol, params_key)
        data_path = self._data_path_regular(interface, symbol, params_key)
        
        # 判断是否需要刷新（优先使用配置的固定间隔，其次使用自适应策略）
        should_refresh_configured, is_configured = self._should_refresh_configured(interface, symbol, params_key)
        
        # 如果配置了固定间隔，使用配置的逻辑
        if is_configured:
            if not should_refresh_configured and data_path.exists():
                logger.info(f"Using cached data for {interface}:{symbol} (configured interval)")
                return pd.read_parquet(data_path)
            # 配置要求刷新或没有缓存，直接下载新数据（跳过后面的自适应检查）
        else:
            # 没有配置，使用自适应策略检查
            if not self._should_refresh_cache(interface, symbol, params_key):
                if data_path.exists():
                    logger.info(f"Using cached data for {interface}:{symbol} (adaptive)")
                    return pd.read_parquet(data_path)
        
        # 下载新数据
        func = getattr(ak, interface)
        try:
            df = func(**kwargs)
        except Exception as e:
            logger.error(f"AkShare接口调用失败: {interface} 参数: {kwargs} 错误: {e}")
            # 网络请求失败，尝试返回缓存数据
            if data_path.exists():
                try:
                    logger.warning(f"Network request failed for {interface}:{symbol}, using cached data")
                    return pd.read_parquet(data_path)
                except Exception as cache_e:
                    logger.warning(f"Failed to load fallback cached data {data_path}: {cache_e}")
            # 如果是OSS存储，尝试从OSS下载
            if self.storage_config.get('type') == 'oss' and not data_path.exists():
                oss_key = self._get_oss_key(str(data_path.relative_to(self.base_path)))
                if self._exists_in_oss(oss_key):
                    if self._download_from_oss(oss_key, data_path):
                        try:
                            logger.info(f"Using cached data from OSS for {interface}:{symbol}")
                            return pd.read_parquet(data_path)
                        except Exception as oss_e:
                            logger.warning(f"Failed to load OSS cached data {data_path}: {oss_e}")
            # 返回带错误信息的DataFrame
            return pd.DataFrame({"error": [f"AkShare接口调用失败: {e}"], "interface": [interface], "params": [str(kwargs)]})
        
        if not df.empty:
            data_path.parent.mkdir(parents=True, exist_ok=True)
            # 确保数据类型兼容Parquet格式
            df = self._normalize_dataframe_for_parquet(df)
            # 使用原子写入确保并发安全
            if not self._write_parquet_with_lock(data_path, df):
                logger.warning(f"Failed to write parquet file for {interface}:{symbol}")
                return df  # 返回数据但不缓存
            
            meta = {
                'interface': interface,
                'symbol': symbol,
                'params_key': params_key,
                'file_path': str(data_path.relative_to(self.base_path)),
                'created_at': datetime.now().isoformat(),
                'data_hash': str(hashlib.md5(df.to_csv(index=False).encode('utf-8')).hexdigest())
            }
            meta_path = self._meta_path_regular(interface, symbol, params_key)
            if not self._write_meta_with_lock(meta_path, meta):
                logger.warning(f"Failed to write meta file for {interface}:{symbol}")
            
            # 更新自适应缓存策略
            self._adaptive_cache_strategy(interface, symbol, df, params_key, **kwargs)
            
            return df
        
        # 下载失败，尝试返回旧缓存
        if data_path.exists():
            try:
                logger.warning(f"Download failed for {interface}:{symbol}, using cached data")
                return pd.read_parquet(data_path)
            except Exception as e:
                logger.warning(f"Failed to load fallback cached data {data_path}: {e}")
        
        return pd.DataFrame({"error": ["AkShare接口无数据且本地无缓存"], "interface": [interface], "params": [str(kwargs)]})
    # 通用参数过滤（可根据实际接口参数分析调整）
    # EXCLUDED_PARAMS = {
    #     'timeout', 'url'  # 只排除通用非业务参数
    # }

    def _get_params_key(self, params: Dict, interface_type: str = 'regular') -> str:
        """
        生成缓存key，根据接口类型决定如何过滤参数

        Args:
            params: 参数字典
            interface_type: 接口类型 ('interval', 'single', 'regular')

        Returns:
            排序后的参数字符串
        """
        if self.param_filter:
            return self.param_filter.get_params_key(params, interface_type)
        else:
            # Fallback to original logic if param_filter not available
            if interface_type == 'interval':
                # 区间接口：过滤掉时间相关参数（因为时间参数由文件名中的start/end区分）
                time_params = {
                    'date', 'start_date', 'end_date', 'trade_date', 'start_time', 'end_time',
                    'start_year', 'end_year', 'year', 'start_page', 'end_page', 'from_page', 'to_page',
                    'from_date', 'to_date', 'start_month', 'end_month', 'from_year', 'to_year'
                }
                filtered = {k: v for k, v in params.items() if k not in {'timeout', 'url'} and k not in time_params}
            elif interface_type == 'single':
                # 单值接口：保留所有参数（时间参数由文件名区分）
                filtered = {k: v for k, v in params.items() if k not in {'timeout', 'url'}}
            else:
                # 普通接口：过滤掉时间相关参数
                time_params = {
                    'date', 'start_date', 'end_date', 'trade_date', 'start_time', 'end_time',
                    'start_year', 'end_year', 'year', 'start_page', 'end_page', 'from_page', 'to_page'
                }
                filtered = {k: v for k, v in params.items() if k not in {'timeout', 'url'} and k not in time_params}

            # 返回更简单的参数键格式，避免特殊字符
            if filtered:
                param_str = '_'.join(f"{k}={v}" for k, v in sorted(filtered.items()))
                return param_str
            else:
                return '[]'

    def _meta_path(self, interface: str, symbol: str, params_key: str, start: str, end: str) -> Path:
        """生成元数据 JSON 路径（本地模式）"""
        fname = f"{interface}_{symbol}_{params_key}_{start}_{end}.meta.json"
        return self.base_path / interface / symbol / fname

    def _data_path(self, interface: str, symbol: str, params_key: str, start: str, end: str) -> Path:
        """生成数据分片 Parquet 路径（本地模式）"""
        fname = f"{interface}_{symbol}_{params_key}_{start}_{end}.parquet"
        return self.base_path / interface / symbol / fname

    def _load_all_metadata(self, interface: str, symbol: str, params_key: str) -> List[dict]:
        """加载所有分片元数据（本地模式）"""
        storage_type = self.storage_config.get('type', 'local')
        
        if storage_type == 'local':
            # 本地存储模式：从本地文件系统加载
            meta_dir = self.base_path / interface / symbol
            if not meta_dir.exists():
                meta_dir.mkdir(parents=True, exist_ok=True)
            
            metadata_list = []
            
            # 先收集所有文件，避免在循环中获取锁
            meta_files = list(meta_dir.glob(f"{interface}_{symbol}_{params_key}_*.meta.json"))
            
            for meta_file in meta_files:
                try:
                    # 尝试直接读取，如果文件被锁定则跳过（非阻塞方式）
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        metadata_list.append(json.load(f))
                except (OSError, json.JSONDecodeError) as e:
                    # 文件被锁定或其他错误，跳过这个文件
                    logger.debug(f"Skipping locked or corrupted meta file {meta_file}: {e}")
                    continue
            
            return metadata_list
            
        elif storage_type == 'oss':
            # OSS存储模式：从OSS加载所有相关的meta文件
            metadata_list = []
            try:
                # 列出OSS上所有相关的meta文件
                prefix = f"{interface}/{symbol}/{interface}_{symbol}_{params_key}_"
                result = self.oss.list_objects(prefix=prefix)
                if result.object_list:
                    for obj in result.object_list:
                        if obj.key.endswith('.meta.json'):
                            local_meta_path = self.base_path / obj.key
                            if self._download_from_oss(obj.key, local_meta_path):
                                logger.debug(f"Downloaded meta file from OSS: {obj.key}")
                                try:
                                    with open(local_meta_path, 'r', encoding='utf-8') as f:
                                        metadata_list.append(json.load(f))
                                except Exception as e:
                                    logger.warning(f"Failed to parse downloaded meta file {local_meta_path}: {e}")
                                    continue
            except Exception as e:
                logger.warning(f"Failed to list/download meta files from OSS: {e}")
            
            return metadata_list
        else:
            logger.error(f"Unsupported storage type: {storage_type}")
            return []

    def _merge_ranges(self, ranges: List[Tuple[str, str]], fmt: str = None) -> List[Tuple[str, str]]:
        """合并重叠区间，支持日期和非日期区间（如分页）"""
        if not ranges or not isinstance(ranges, list):
            return []
        
        # 过滤掉无效的区间
        valid_ranges = []
        for r in ranges:
            if isinstance(r, (list, tuple)) and len(r) == 2 and r[0] and r[1]:
                valid_ranges.append((str(r[0]), str(r[1])))
        
        if not valid_ranges:
            return []
            
        if fmt and fmt not in ("page", "time", None):
            # 日期型区间，按字符串排序即可（假定格式如%Y%m%d）
            try:
                valid_ranges = sorted(valid_ranges, key=lambda x: x[0])
            except Exception as e:
                logger.warning(f"Failed to sort date ranges: {e}")
                return valid_ranges
        else:
            # 非日期区间（如分页），直接按int排序
            try:
                valid_ranges = sorted(valid_ranges, key=lambda x: int(x[0]))
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to sort numeric ranges: {e}")
                return valid_ranges
                
        merged = [valid_ranges[0]]
        for start, end in valid_ranges[1:]:
            last_start, last_end = merged[-1]
            # 日期区间直接字符串比较，分页区间用int比较
            try:
                if fmt and fmt not in ("page", "time", None):
                    overlap = start <= last_end
                else:
                    overlap = int(start) <= int(last_end)
                    
                if overlap:
                    if fmt and fmt not in ("page", "time", None):
                        merged[-1] = (last_start, max(last_end, end))
                    else:
                        merged[-1] = (str(min(int(last_start), int(start))), str(max(int(last_end), int(end))))
                else:
                    merged.append((start, end))
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to merge range ({start}, {end}): {e}")
                merged.append((start, end))
                
        return merged

    def _get_file_size_mb(self, file_path: Path) -> float:
        """获取文件大小（MB）"""
        if file_path.exists():
            return file_path.stat().st_size / (1024 * 1024)
        return 0.0

    def _should_merge_small_files(self, interface: str, symbol: str, params_key: str, 
                                cached_ranges: List[Tuple[str, str]], fmt: str) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        """
        检查是否需要合并小文件
        
        Args:
            interface: 接口名
            symbol: 股票代码
            params_key: 参数键
            cached_ranges: 已缓存的区间列表
            fmt: 格式类型
            
        Returns:
            需要合并的区间对列表 [(range1, range2), ...]
        """
        if not cached_ranges or len(cached_ranges) < 2:
            return []
        
        merge_pairs = []
        sorted_ranges = sorted(cached_ranges, key=lambda x: x[0])
        
        for i in range(len(sorted_ranges) - 1):
            current_range = sorted_ranges[i]
            next_range = sorted_ranges[i + 1]
            
            # 检查两个区间是否相邻
            is_adjacent = False
            if fmt and fmt not in ("page", "time", None):
                # 日期区间：检查是否相邻（相差1天）
                try:
                    from datetime import datetime, timedelta
                    current_end = datetime.strptime(current_range[1], fmt).date()
                    next_start = datetime.strptime(next_range[0], fmt).date()
                    is_adjacent = (next_start - current_end).days <= 1
                except:
                    is_adjacent = False
            else:
                # 分页区间：检查是否连续
                try:
                    is_adjacent = int(next_range[0]) - int(current_range[1]) <= 1
                except:
                    is_adjacent = False
            
            if is_adjacent:
                # 检查两个文件的大小
                current_path = self._data_path(interface, symbol, params_key, current_range[0], current_range[1])
                next_path = self._data_path(interface, symbol, params_key, next_range[0], next_range[1])
                
                current_size = self._get_file_size_mb(current_path)
                next_size = self._get_file_size_mb(next_path)
                
                # 如果任一文件小于阈值，建议合并
                if current_size < self.min_file_size_mb or next_size < self.min_file_size_mb:
                    merge_pairs.append((current_range, next_range))
        
        return merge_pairs

    def _merge_adjacent_files(self, interface: str, symbol: str, params_key: str,
                            range1: Tuple[str, str], range2: Tuple[str, str], 
                            keys: tuple, fmt: str) -> bool:
        """
        合并两个相邻的文件
        
        Args:
            interface: 接口名
            symbol: 股票代码
            params_key: 参数键
            range1: 第一个区间
            range2: 第二个区间
            keys: 参数键元组
            fmt: 格式类型
            
        Returns:
            bool: 合并是否成功
        """
        try:
            # 加载两个文件的数据
            path1 = self._data_path(interface, symbol, params_key, range1[0], range1[1])
            path2 = self._data_path(interface, symbol, params_key, range2[0], range2[1])
            
            if not path1.exists() or not path2.exists():
                logger.warning(f"Files not found for merging: {path1.exists()}, {path2.exists()}")
                return False
            
            df1 = pd.read_parquet(path1)
            df2 = pd.read_parquet(path2)
            
            # 合并数据
            merged_df = pd.concat([df1, df2], ignore_index=True)
            
            # 去重和排序
            for col in ['date', 'trade_date', keys[0], keys[1]]:
                if col in merged_df.columns:
                    merged_df = merged_df.drop_duplicates(subset=[col]).sort_values(col)
            
            merged_df = merged_df.reset_index(drop=True)
            
            # 创建新的合并区间
            if fmt and fmt not in ("page", "time", None):
                # 日期区间
                merged_start = min(range1[0], range2[0])
                merged_end = max(range1[1], range2[1])
            else:
                # 分页区间
                merged_start = str(min(int(range1[0]), int(range2[0])))
                merged_end = str(max(int(range1[1]), int(range2[1])))
            
            # 保存合并后的文件
            merged_path = self._data_path(interface, symbol, params_key, merged_start, merged_end)
            merged_path.parent.mkdir(parents=True, exist_ok=True)
            # 确保数据类型兼容Parquet格式
            merged_df = self._normalize_dataframe_for_parquet(merged_df)
            # 使用原子写入确保并发安全
            if not self._write_parquet_with_lock(merged_path, merged_df):
                logger.warning(f"Failed to write merged parquet file for {interface}:{symbol}")
                return False
            
            # 创建新的meta文件
            meta_path = self._meta_path(interface, symbol, params_key, merged_start, merged_end)
            meta = {
                'interface': interface,
                'symbol': symbol,
                'params_key': params_key,
                keys[0]: merged_start,
                keys[1]: merged_end,
                'file_path': str(merged_path.relative_to(self.base_path)),
                'created_at': datetime.now().isoformat(),
                'data_hash': str(hashlib.md5(merged_df.to_csv(index=False).encode('utf-8')).hexdigest()),
                'merged_from': [range1, range2]  # 记录合并来源
            }
            
            if not self._write_meta_with_lock(meta_path, meta):
                logger.warning(f"Failed to write merged meta file for {interface}:{symbol}")
                # 清理合并文件
                merged_path.unlink(missing_ok=True)
                return False
            
            # 删除旧文件
            logger.info(f"Deleting old files: {path1}, {path2}")
            path1.unlink(missing_ok=True)
            path2.unlink(missing_ok=True)
            
            # 删除旧的meta文件
            meta_path1 = self._meta_path(interface, symbol, params_key, range1[0], range1[1])
            meta_path2 = self._meta_path(interface, symbol, params_key, range2[0], range2[1])
            logger.info(f"Deleting old meta files: {meta_path1}, {meta_path2}")
            meta_path1.unlink(missing_ok=True)
            meta_path2.unlink(missing_ok=True)
            
            logger.info(f"Merged small files for {interface}:{symbol}: {range1} + {range2} -> ({merged_start}, {merged_end})")
            return True
            
        except Exception as e:
            logger.error(f"Error merging files {range1} and {range2}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _find_missing_ranges(self, req_start: str, req_end: str, cached_ranges: List[Tuple[str, str]], fmt: str = None) -> List[Tuple[str, str]]:
        """找出缺失区间，支持日期和非日期区间（如分页）"""
        if fmt and fmt not in ("page", "time", None):
            from datetime import datetime, timedelta

            # 支持多种日期格式
            def parse_date(date_str):
                """灵活的日期解析"""
                date_formats = ['%Y%m%d', '%Y-%m-%d', '%Y/%m/%d']
                for fmt in date_formats:
                    try:
                        return datetime.strptime(date_str, fmt).date()
                    except ValueError:
                        continue
                raise ValueError(f"Unsupported date format: {date_str}")

            def str2d(s): return parse_date(s)
            def d2str(d): return d.strftime('%Y%m%d')  # 统一输出格式

            req_s, req_e = str2d(req_start), str2d(req_end)
            if not cached_ranges:
                return [(req_start, req_end)]  # 返回原始格式
            merged = self._merge_ranges(cached_ranges, fmt)
            missing = []
            cur = req_s
            for s, e in merged:
                s, e = str2d(s), str2d(e)
                if cur < s:
                    miss_e = min(req_e, s - timedelta(days=1))
                    if cur <= miss_e:
                        missing.append((d2str(cur), d2str(miss_e)))
                cur = max(cur, e + timedelta(days=1))
                if cur > req_e:
                    break
            if cur <= req_e:
                missing.append((d2str(cur), d2str(req_e)))
            return missing
        else:
            # 非日期区间（如分页）
            req_s, req_e = int(req_start), int(req_end)
            if not cached_ranges:
                return [(str(req_start), str(req_end))]
            merged = self._merge_ranges(cached_ranges, fmt)
            missing = []
            cur = req_s
            for s, e in merged:
                s, e = int(s), int(e)
                if cur < s:
                    miss_e = min(req_e, s - 1)
                    if cur <= miss_e:
                        missing.append((str(cur), str(miss_e)))
                cur = max(cur, e + 1)
                if cur > req_e:
                    break
            if cur <= req_e:
                missing.append((str(cur), str(req_e)))
            return missing

    def get_data(self, interface: str, symbol: str = '', **kwargs) -> pd.DataFrame:
        """
        统一入口：根据接口类型自动分流
        - 区间/分页接口：自动识别所有常见 AkShare 区间参数组合，走分片增量逻辑
        - 单date/period接口：直接走单分片逻辑
        - 其它非时间序列接口：走自适应缓存逻辑
        
        支持特殊参数：
        - force_refresh: bool, 强制刷新缓存
        """
        # 处理特殊参数
        force_refresh = kwargs.pop('force_refresh', False)  # 强制刷新
        
        # 处理symbol/code参数兼容性：如果symbol为空但有code参数，用code作为symbol
        if not symbol and 'code' in kwargs:
            symbol = str(kwargs['code'])
        
        # 处理特定接口的默认参数
        if interface == 'stock_main_fund_flow' and not symbol:
            symbol = '全部股票'  # 默认值
        
        # 区间参数组合（start/end）
        interval_patterns = [
            (('start_date', 'end_date'), '%Y%m%d'),
            (('start_year', 'end_year'), '%Y'),
            (('start_month', 'end_month'), '%Y%m'),
            (('from_date', 'to_date'), '%Y%m%d'),
            (('from_year', 'to_year'), '%Y'),
            (('from_page', 'to_page'), 'page'),
            (('start_time', 'end_time'), 'time'),
           
        ]
        # 单date/period参数
        single_patterns = [
            (('period',), None),
            (('date',), None),
            (('trade_date',), None),
            (('year',), None),
        ]
        
        # 强制刷新：跳过缓存，直接调用接口
        if force_refresh:
            logger.info(f"Force refreshing cache for {interface}:{symbol}")
            return self._download_fresh_data(interface, symbol, **kwargs)
        
        # 优先区间分支
        for keys, fmt in interval_patterns:
            if all(k in kwargs for k in keys):
                return self._handle_interval(interface, symbol, keys, fmt, kwargs)
        # 单date/period分支
        for keys, _ in single_patterns:
            if all(k in kwargs for k in keys):
                return self._handle_single(interface, symbol, keys, kwargs)
        # 其它分支
        params_key = self._get_params_key(kwargs, 'regular')
        return self._handle_regular(interface, symbol, params_key, kwargs)
    
    def _download_fresh_data(self, interface: str, symbol: str, **kwargs) -> pd.DataFrame:
        """
        强制刷新：直接调用接口下载最新数据，不使用缓存
        """
        func = getattr(ak, interface)
        try:
            df = func(**kwargs)
            logger.info(f"Downloaded fresh data for {interface}:{symbol} ({len(df)} records)")
            return df
        except Exception as e:
            logger.warning(f"AkShare接口调用失败: {interface} {kwargs}: {e}")
            return pd.DataFrame()
    def _handle_interval(self, interface: str, symbol: str, keys, fmt, kwargs) -> pd.DataFrame:
        """
        区间/分页接口分片主流程（start/end参数），每个分片都缓存，缺失分片自动补全，最终合并所有分片。
        """
        # 验证日期参数
        start_val = kwargs.get(keys[0])
        end_val = kwargs.get(keys[1])
        
        if start_val is None or str(start_val).strip() == '':
            raise ValueError(f"Required parameter '{keys[0]}' cannot be None or empty")
        if end_val is None or str(end_val).strip() == '':
            raise ValueError(f"Required parameter '{keys[1]}' cannot be None or empty")
        
        start, end = str(start_val), str(end_val)
        params_key = self._get_params_key(kwargs, 'interval')
        meta_list = self._load_all_metadata(interface, symbol, params_key)
        # 兼容不同区间参数名
        cached_ranges = []
        for m in meta_list:
            s = m.get(keys[0]) or m.get('start_date') or m.get('from_date') or m.get('start_year') or m.get('from_page')
            e = m.get(keys[1]) or m.get('end_date') or m.get('to_date') or m.get('end_year') or m.get('to_page')
            if s is not None and e is not None:
                cached_ranges.append((str(s), str(e)))
        merged = self._merge_ranges(cached_ranges, fmt)
        # 找出缺失分片
        missing = self._find_missing_ranges(start, end, merged, fmt)
        # 下载缺失分片并缓存
        for miss_start, miss_end in missing:
            part_kwargs = dict(kwargs)
            part_kwargs[keys[0]] = miss_start
            part_kwargs[keys[1]] = miss_end
            try:
                func = getattr(ak, interface)
                df = func(**part_kwargs)
            except Exception as e:
                logger.warning(f"AkShare接口调用失败: {interface} {part_kwargs}: {e}")
                df = pd.DataFrame()
            if not df.empty:
                data_path = self._data_path(interface, symbol, params_key, miss_start, miss_end)
                meta_path = self._meta_path(interface, symbol, params_key, miss_start, miss_end)
                data_path.parent.mkdir(parents=True, exist_ok=True)
                # 确保数据类型兼容Parquet格式
                df = self._normalize_dataframe_for_parquet(df)
                # 使用原子写入确保并发安全
                if not self._write_parquet_with_lock(data_path, df):
                    logger.warning(f"Failed to write parquet file for {interface}:{symbol} interval {miss_start}-{miss_end}")
                    continue  # 跳过这个分片
                
                meta = {
                    'interface': interface,
                    'symbol': symbol,
                    'params_key': params_key,
                    keys[0]: miss_start,
                    keys[1]: miss_end,
                    'file_path': str(data_path.relative_to(self.base_path)),
                    'created_at': datetime.now().isoformat(),
                    'data_hash': str(hashlib.md5(df.to_csv(index=False).encode('utf-8')).hexdigest())
                }
                if not self._write_meta_with_lock(meta_path, meta):
                    logger.warning(f"Failed to write meta file for {interface}:{symbol} interval {miss_start}-{miss_end}")
        
        # 合并所有分片（包括新下载和已有缓存），添加内存限制
        all_dfs = []
        total_memory_mb = 0
        max_memory_mb = 500  # 限制最大内存使用500MB
        
        for m in self._load_all_metadata(interface, symbol, params_key):
            data_path = self.base_path / m['file_path']
            df = None
            
            if data_path.exists():
                try:
                    df = pd.read_parquet(data_path)
                except Exception as e:
                    logger.warning(f"Failed to load cached interval data {data_path}: {e}")
                    continue
            
            # 如果成功加载了数据，进行内存检查和日期过滤
            if df is not None:
                df_memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                
                # 检查内存限制
                if total_memory_mb + df_memory_mb > max_memory_mb:
                    logger.warning(f"Memory limit exceeded ({total_memory_mb + df_memory_mb:.1f}MB > {max_memory_mb}MB), "
                                 f"skipping remaining data files")
                    break
                
                total_memory_mb += df_memory_mb
                
                # 只保留请求区间内的数据
                for col in ['date', 'trade_date', keys[0], keys[1]]:
                    if col in df.columns:
                        if fmt and fmt != 'page':
                            # 处理日期比较问题：确保类型一致
                            try:
                                if df[col].dtype == 'object':
                                    # 如果是字符串类型，转换为日期进行比较
                                    df = df[(pd.to_datetime(df[col]) >= pd.to_datetime(start)) &
                                           (pd.to_datetime(df[col]) <= pd.to_datetime(end))]
                                else:
                                    # 如果已经是日期类型，直接比较
                                    df = df[(df[col] >= pd.to_datetime(start)) &
                                           (df[col] <= pd.to_datetime(end))]
                            except Exception as e:
                                logger.warning(f"Date filtering failed for column {col}: {e}")
                                # 如果转换失败，保持原数据
                                break
                
                all_dfs.append(df)
        if all_dfs:
            result = pd.concat(all_dfs, ignore_index=True)
            # 按主时间字段去重排序
            for col in ['date', 'trade_date', keys[0], keys[1]]:
                if col in result.columns:
                    result = result.drop_duplicates(subset=[col]).sort_values(col)
            result = result.reset_index(drop=True)
            return result
        # 没有任何分片数据，返回空DataFrame
        return pd.DataFrame()

    def _handle_single(self, interface: str, symbol: str, keys, kwargs) -> pd.DataFrame:
        """
        单date/period参数接口主流程（如date、trade_date、year、period）
        优先命中本地缓存，只有无缓存或明确需要刷新时才请求接口，接口失败兜底返回缓存。
        """
        params_key = self._get_params_key(kwargs, 'single')
        
        # 跟踪使用频率
        self._track_usage(interface, symbol, params_key)
        data_path = self._data_path_regular(interface, symbol, params_key)
        
        # 首先尝试从OSS下载（如果配置了OSS且本地没有缓存）
        if self.storage_config.get('type') == 'oss' and not data_path.exists():
            oss_key = self._get_oss_key(str(data_path.relative_to(self.base_path)))
            if self._exists_in_oss(oss_key):
                if self._download_from_oss(oss_key, data_path):
                    logger.debug(f"Downloaded data from OSS for {interface}:{symbol}")
        
        # 优先读缓存（只要缓存存在且未过期）
        if data_path.exists() and not self._should_refresh_cache(interface, symbol, params_key):
            try:
                return pd.read_parquet(data_path)
            except Exception as e:
                logger.warning(f"Failed to load cached data {data_path}: {e}")
        
        # 需要刷新或无缓存，尝试请求接口
        func = getattr(ak, interface)
        try:
            df = func(**kwargs)
        except Exception as e:
            logger.warning(f"AkShare接口调用失败: {interface} {kwargs}: {e}")
            df = pd.DataFrame()
        if not df.empty:
            data_path.parent.mkdir(parents=True, exist_ok=True)
            # 确保数据类型兼容Parquet格式
            df = self._normalize_dataframe_for_parquet(df)
            # 使用原子写入确保并发安全
            if not self._write_parquet_with_lock(data_path, df):
                logger.warning(f"Failed to write parquet file for {interface}:{symbol} single")
                return df  # 返回数据但不缓存
            
            meta = {
                'interface': interface,
                'symbol': symbol,
                'params_key': params_key,
                'file_path': str(data_path.relative_to(self.base_path)),
                'created_at': datetime.now().isoformat(),
                'data_hash': str(hashlib.md5(df.to_csv(index=False).encode('utf-8')).hexdigest())
            }
            meta_path = self._meta_path_regular(interface, symbol, params_key)
            if not self._write_meta_with_lock(meta_path, meta):
                logger.warning(f"Failed to write meta file for {interface}:{symbol} single")
            
            # 更新自适应缓存策略
            self._adaptive_cache_strategy(interface, symbol, df, params_key, **kwargs)
            
            return df
        # 接口无数据或失败，兜底返回本地缓存
        if data_path.exists():
            try:
                return pd.read_parquet(data_path)
            except Exception as e:
                logger.warning(f"Failed to load fallback cached data {data_path}: {e}")
        return pd.DataFrame()
    def __init__(self, storage_config: Dict[str, Any]):
        """
        Args:
            storage_config: {'type': 'local'|'oss', 'path': './cache', 'min_file_size_mb': 1.0, ...}
        """
        self.storage_config = storage_config
        self.min_file_size_mb = storage_config.get('min_file_size_mb', 1.0)  # 小文件合并阈值（MB）

        # 初始化优化组件
        self._init_optimization_components()

        self._init_storage()

    def _init_optimization_components(self):
        """初始化优化组件"""
        if _has_optimization:
            try:
                self.config_manager = get_config_manager()
                self.param_filter = get_parameter_filter()
                logger.info("Optimization components initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize optimization components: {e}")
                self.config_manager = None
                self.param_filter = None
        else:
            self.config_manager = None
            self.param_filter = None

    def _init_storage(self):
        """初始化存储后端"""
        stype = self.storage_config.get('type', 'local')

        # 检查OSS配置是否完整
        if stype == 'oss':
            required_oss_params = ['access_key_id', 'access_key_secret', 'endpoint', 'bucket']
            oss_params_present = all(param in self.storage_config for param in required_oss_params)

            if not oss_params_present:
                logger.warning("OSS配置不完整，将回退到本地存储模式")
                logger.warning("OSS需要以下参数: access_key_id, access_key_secret, endpoint, bucket")
                stype = 'local'  # 回退到本地存储
                # 当OSS回退到本地时，使用local_cache_path作为本地路径
                if 'local_cache_path' in self.storage_config:
                    self.storage_config['path'] = self.storage_config['local_cache_path']

        if stype == 'local':
            self.base_path = Path(self.storage_config.get('path', './cache'))
            self.base_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"使用本地存储模式，缓存路径: {self.base_path}")
        elif stype == 'oss':
            oss2 = _get_oss2()
            if oss2:
                try:
                    # 先创建Auth对象
                    auth = oss2.Auth(
                        self.storage_config['access_key_id'],
                        self.storage_config['access_key_secret']
                    )
                    # 再创建Bucket对象
                    self.oss = oss2.Bucket(auth, self.storage_config['endpoint'], self.storage_config['bucket'])
                    # 为OSS设置本地缓存路径
                    self.base_path = Path(self.storage_config.get('local_cache_path', './cache'))
                    self.base_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"使用阿里云OSS存储模式，存储桶: {self.storage_config['bucket']}")
                except Exception as e:
                    logger.error(f"OSS连接失败: {e}，将回退到本地存储模式")
                    self.base_path = Path(self.storage_config.get('path', './cache'))
                    self.base_path.mkdir(parents=True, exist_ok=True)
                    # 确保清理任何可能的云存储属性
                    if hasattr(self, 'oss'):
                        delattr(self, 'oss')
            else:
                logger.error("oss2包不可用，将回退到本地存储模式")
                self.base_path = Path(self.storage_config.get('path', './cache'))
                self.base_path.mkdir(parents=True, exist_ok=True)
                # 确保清理任何可能的云存储属性
                if hasattr(self, 'oss'):
                    delattr(self, 'oss')
        else:
            logger.warning(f"不支持的存储类型: {stype}，将使用本地存储模式")
            self.base_path = Path(self.storage_config.get('path', './cache'))
            self.base_path.mkdir(parents=True, exist_ok=True)
            # 确保清理任何可能的云存储属性
            if hasattr(self, 'oss'):
                delattr(self, 'oss')

    def _upload_to_oss(self, local_path: Path, oss_key: str) -> bool:
        """
        原子上传文件到阿里云OSS（使用分块上传确保原子性）

        Args:
            local_path: 本地文件路径
            oss_key: OSS对象键

        Returns:
            bool: 上传是否成功
        """
        if self.storage_config.get('type') != 'oss':
            logger.debug(f"OSS not configured, skipping upload of {local_path}")
            return False

        try:
            # 获取文件大小
            file_size = local_path.stat().st_size
            
            # 对于小文件，直接上传
            if file_size <= 100 * 1024 * 1024:  # 100MB以下直接上传
                with open(local_path, 'rb') as f:
                    self.oss.put_object(oss_key, f)
                logger.info(f"Successfully uploaded {local_path} to OSS as {oss_key}")
                return True
            
            # 对于大文件，使用分块上传确保原子性
            # 1. 初始化分块上传
            upload_id = self.oss.init_multipart_upload(oss_key).upload_id
            
            # 2. 分块上传
            parts = []
            part_size = 100 * 1024 * 1024  # 100MB per part
            part_number = 1
            
            with open(local_path, 'rb') as f:
                while True:
                    data = f.read(part_size)
                    if not data:
                        break
                    
                    # 上传分块
                    result = self.oss.upload_part(oss_key, upload_id, part_number, data)
                    parts.append({
                        'ETag': result.etag,
                        'PartNumber': part_number
                    })
                    part_number += 1
            
            # 3. 完成分块上传（原子操作）
            self.oss.complete_multipart_upload(oss_key, upload_id, parts)
            
            logger.info(f"Successfully uploaded large file {local_path} to OSS as {oss_key} using multipart upload")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to OSS {oss_key}: {e}")
            # 尝试清理失败的分块上传
            try:
                if 'upload_id' in locals():
                    self.oss.abort_multipart_upload(oss_key, upload_id)
            except:
                pass
            return False

    def _download_from_oss(self, oss_key: str, local_path: Path) -> bool:
        """
        从阿里云OSS下载文件

        Args:
            oss_key: OSS对象键
            local_path: 本地文件路径

        Returns:
            bool: 下载是否成功
        """
        if self.storage_config.get('type') != 'oss':
            return False

        try:
            # 确保本地目录存在
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # 从OSS下载文件
            obj = self.oss.get_object(oss_key)
            with open(local_path, 'wb') as f:
                f.write(obj.read())
            logger.debug(f"Successfully downloaded {oss_key} from OSS to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {oss_key} from OSS: {e}")
            return False

    def _exists_in_oss(self, oss_key: str) -> bool:
        """
        检查OSS上是否存在文件

        Args:
            oss_key: OSS对象键

        Returns:
            bool: 文件是否存在
        """
        if self.storage_config.get('type') != 'oss':
            return False

        try:
            # 检查对象是否存在
            return self.oss.object_exists(oss_key)
        except Exception as e:
            logger.debug(f"Error checking existence of {oss_key} in OSS: {e}")
            return False

    def _get_oss_key(self, relative_path: str) -> str:
        """
        将相对路径转换为OSS键

        Args:
            relative_path: 相对于base_path的路径

        Returns:
            str: OSS对象键
        """
        # 移除开头的./或/，并用/替换路径分隔符
        oss_key = relative_path.lstrip('./').lstrip('/')
        return oss_key.replace('\\', '/')  # 确保使用正斜杠

    def _track_usage(self, interface: str, symbol: str, params_key: str):
        """跟踪接口使用频率 - 写入到对应的meta文件中"""
        # 使用对应的meta文件
        meta_path = self._meta_path_regular(interface, symbol, params_key)
        
        # 使用文件锁确保并发安全
        fd = self._acquire_lock(meta_path, timeout=5)
        if fd is None:
            logger.warning("Failed to acquire lock for usage tracking")
            return
        
        try:
            now = datetime.now().isoformat()
            
            # 加载现有meta数据
            meta_data = {}
            if meta_path.exists():
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Failed to load meta file, starting fresh: {e}")
                    meta_data = {}
            
            # 初始化usage统计（如果不存在）
            if 'usage_stats' not in meta_data:
                meta_data['usage_stats'] = {
                    'access_count': 0,
                    'last_access': now,
                    'priority_score': 0
                }
            
            # 确保meta文件包含基本信息
            if 'interface' not in meta_data:
                meta_data['interface'] = interface
            if 'symbol' not in meta_data:
                meta_data['symbol'] = symbol
            if 'params_key' not in meta_data:
                meta_data['params_key'] = params_key or ''
            
            # 更新统计
            stats = meta_data['usage_stats']
            stats['access_count'] += 1
            stats['last_access'] = now
            
            # 计算优先级分数（基于访问频率和时间衰减）
            last_access_dt = datetime.fromisoformat(stats['last_access'])
            days_since_last = (datetime.now() - last_access_dt).days
            stats['priority_score'] = stats['access_count'] * (1.0 / (1 + days_since_last))
            
            # 保存meta数据（在同一个锁的保护下）
            if not self._write_meta_with_lock(meta_path, meta_data):
                logger.warning("Failed to write usage statistics to meta file")
        
        except Exception as e:
            logger.error(f"Error in usage tracking: {e}")
        finally:
            self._release_lock(fd)

    def _adaptive_cache_strategy(self, interface: str, symbol: str, current_data: pd.DataFrame,
                                cache_key: str, **kwargs) -> str:
        """
        自适应缓存策略：基于数据更新频率动态调整缓存时间

        Args:
            interface: 接口名
            symbol: 股票代码
            current_data: 当前获取的数据
            cache_key: 缓存键
            **kwargs: 其他参数

        Returns:
            str: 下次建议的缓存过期时间（ISO格式）
        """
        adaptive_path = self.base_path / "adaptive_cache.json"
        
        # 使用文件锁确保并发安全
        fd = self._acquire_lock(adaptive_path, timeout=5)
        if fd is None:
            logger.warning("Failed to acquire lock for adaptive cache")
            return (datetime.now() + timedelta(minutes=15)).isoformat()
        
        try:
            # 计算当前数据的哈希值（使用更稳定的方法）
            # 对DataFrame进行规范化：排序列名，按行排序，确保一致性
            df_copy = current_data.copy()
            df_copy = df_copy.sort_index(axis=1)  # 按列名排序
            if not df_copy.empty and len(df_copy) > 0:
                df_copy = df_copy.sort_values(by=df_copy.columns[0])  # 按第一列排序
            data_str = df_copy.to_csv(index=False)
            data_hash = hashlib.md5(data_str.encode('utf-8')).hexdigest()
            
            # 加载现有自适应缓存数据
            adaptive_cache = {}
            if adaptive_path.exists():
                try:
                    with open(adaptive_path, 'r', encoding='utf-8') as f:
                        adaptive_cache = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Failed to load adaptive cache, starting fresh: {e}")
                    adaptive_cache = {}
            
            key = f"{interface}:{symbol}:{cache_key}"
            now = datetime.now()
            
            if key in adaptive_cache:
                prev_data = adaptive_cache[key]
                prev_hash = prev_data.get('data_hash')
                last_check_str = prev_data.get('last_check_time')
                interval_minutes = prev_data.get('check_interval_minutes', 15)
                same_count = prev_data.get('consecutive_same_count', 0)
                last_update_str = prev_data.get('last_update_time')
                
                last_check = datetime.fromisoformat(last_check_str) if last_check_str else now
                last_update = datetime.fromisoformat(last_update_str) if last_update_str else last_check
                
                # 检查数据是否发生变化
                if data_hash == prev_hash:
                    # 数据没有变化，增加连续相同计数
                    same_count += 1
                    
                    # 根据连续相同次数动态调整检查间隔
                    if same_count >= 10:  # 连续10次相同，认为是低频更新
                        new_interval = min(interval_minutes * 2, 1440 * 7)  # 最多7天
                    elif same_count >= 5:  # 连续5次相同，认为是中频更新
                        new_interval = min(interval_minutes * 1.5, 1440)  # 最多1天
                    else:  # 连续几次相同，保持当前间隔
                        new_interval = interval_minutes
                    
                    next_check = now + timedelta(minutes=new_interval)
                    
                else:
                    # 数据发生变化，重置计数和间隔
                    same_count = 0
                    new_interval = max(interval_minutes / 2, 5)  # 最少5分钟
                    next_check = now + timedelta(minutes=new_interval)
                    last_update = now
                    
            else:
                # 首次缓存，设置初始间隔
                same_count = 0
                new_interval = 15  # 初始15分钟检查一次
                next_check = now + timedelta(minutes=new_interval)
                last_update = now
            
            # 更新缓存记录
            adaptive_cache[key] = {
                'interface': interface,
                'symbol': symbol,
                'cache_key': cache_key,
                'data_hash': data_hash,
                'last_check_time': now.isoformat(),
                'check_interval_minutes': new_interval,
                'consecutive_same_count': same_count,
                'last_update_time': last_update.isoformat()
            }
            
            # 保存自适应缓存数据（在同一个锁的保护下）
            if not self._write_meta_with_lock(adaptive_path, adaptive_cache):
                logger.warning("Failed to write adaptive cache data")
            
            logger.info(f"Adaptive cache for {interface}:{symbol} - same_count: {same_count}, "
                       f"next_check_interval: {new_interval}min")
            
            return next_check.isoformat()
        
        except Exception as e:
            logger.error(f"Error in adaptive cache strategy: {e}")
            return (datetime.now() + timedelta(minutes=15)).isoformat()
        finally:
            self._release_lock(fd)

    def _should_refresh_cache(self, interface: str, symbol: str, cache_key: str) -> bool:
        """
        判断是否应该刷新缓存（基于自适应策略）

        Args:
            interface: 接口名
            symbol: 股票代码
            cache_key: 缓存键

        Returns:
            bool: 是否需要刷新
        """
        adaptive_path = self.base_path / "adaptive_cache.json"
        
        # 使用文件锁确保并发安全
        fd = self._acquire_lock(adaptive_path, timeout=5)
        if fd is None:
            logger.warning("Failed to acquire lock for cache refresh check")
            return True  # 获取锁失败时，假设需要刷新
        
        try:
            if not adaptive_path.exists():
                return True  # 没有记录，需要刷新
            
            with open(adaptive_path, 'r', encoding='utf-8') as f:
                adaptive_cache = json.load(f)
            
            key = f"{interface}:{symbol}:{cache_key}"
            if key not in adaptive_cache:
                return True  # 没有记录，需要刷新
            
            data = adaptive_cache[key]
            last_check_str = data.get('last_check_time')
            interval_minutes = data.get('check_interval_minutes', 15)
            
            if not last_check_str:
                return True
            
            last_check = datetime.fromisoformat(last_check_str)
            now = datetime.now()
            
            # 检查是否超过检查间隔
            time_diff = (now - last_check).total_seconds() / 60  # 分钟
            return time_diff >= interval_minutes
        
        except Exception as e:
            logger.error(f"Error checking cache refresh: {e}")
            return True  # 出错时假设需要刷新
        finally:
            self._release_lock(fd)

    def __getattr__(self, name: str):
        """
        通过反射实现优雅的接口调用

        允许用户直接调用 wrapper.stock_zh_index_daily(symbol='000001')
        而不需要 wrapper.get_data('stock_zh_index_daily', symbol='000001')

        这使得从 akshare 迁移到我们的缓存系统时只需要改动一行：
        import akshare as ak  ->  wrapper = AkShareWrapperV2(...)
        ak.xxx(...)          ->  wrapper.xxx(...)
        """
        def _call_akshare_interface(**kwargs):
            # 直接调用 get_data 方法，传入接口名和参数
            return self.get_data(name, **kwargs)

        # 返回这个包装函数
        return _call_akshare_interface
    
    def _write_parquet_with_lock(self, parquet_path: Path, df: pd.DataFrame, timeout: int = 10) -> bool:
        """
        使用文件锁和原子写入安全地写入Parquet文件
        
        Args:
            parquet_path: Parquet文件路径
            df: 要写入的DataFrame
            timeout: 锁超时时间（秒）
            
        Returns:
            bool: 是否写入成功
        """
        storage_type = self.storage_config.get('type', 'local')
        
        if storage_type == 'local':
            # 本地存储模式：只写入本地文件
            # 获取文件锁（单机环境）
            fd = self._acquire_lock(parquet_path, timeout=timeout)
            if fd is None:
                logger.warning(f"Failed to acquire lock for writing parquet file: {parquet_path}")
                return False
            
            try:
                # 使用临时文件写入，然后原子性替换
                temp_path = parquet_path.with_suffix('.tmp')
                
                # 先写入临时文件
                df.to_parquet(temp_path)
                
                # 确保数据写入磁盘
                temp_fd = os.open(str(temp_path), os.O_RDONLY)
                try:
                    os.fsync(temp_fd)
                finally:
                    os.close(temp_fd)
                
                # 原子性替换原文件
                temp_path.replace(parquet_path)
                
                # 确保目录操作同步到磁盘
                dir_fd = os.open(str(parquet_path.parent), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
                
                return True
                
            except Exception as e:
                logger.error(f"Error writing parquet file {parquet_path}: {e}")
                # 清理临时文件
                if temp_path.exists():
                    temp_path.unlink()
                return False
            finally:
                self._release_lock(fd)
                
        elif storage_type == 'oss':
            # OSS存储模式：只上传到OSS，不写入本地文件
            try:
                # 创建临时文件用于上传
                temp_path = parquet_path.with_suffix('.tmp')
                df.to_parquet(temp_path)
                
                # 上传到OSS
                oss_key = self._get_oss_key(str(parquet_path.relative_to(self.base_path)))
                success = self._upload_to_oss(temp_path, oss_key)
                
                # 清理临时文件
                if temp_path.exists():
                    temp_path.unlink()
                
                if success:
                    logger.info(f"Successfully uploaded parquet file to OSS: {oss_key}")
                    return True
                else:
                    logger.error(f"Failed to upload parquet file to OSS: {oss_key}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error uploading parquet file to OSS {parquet_path}: {e}")
                # 清理临时文件
                if temp_path.exists():
                    temp_path.unlink()
                return False
        else:
            logger.error(f"Unsupported storage type: {storage_type}")
            return False

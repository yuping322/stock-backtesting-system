"""state_store: in-memory minimal state for positions, nav, orders and simple persistence hooks
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List
import pandas as pd
import json
import os
import threading
import hashlib
from datetime import datetime
import tempfile
import shutil


@dataclass
class StateStore:
    positions: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["code","shares","avg_price","weight"]))
    orders: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["order_id","code","side","shares","status"]))
    nav: List[Dict[str, Any]] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def snapshot_positions(self) -> pd.DataFrame:
        return self.positions.copy()

    def append_fill(self, fill: Dict[str, Any]):
        # simple append logic
        with self._lock:
            self.nav.append({"ts": fill.get("ts"), "fill": fill})

    def save(self, path: str):
        """Save state to file with atomic write and backup"""
        with self._lock:
            state_data = {
                'positions': self.positions.to_dict('records'),
                'orders': self.orders.to_dict('records'),
                'nav': self.nav,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }

            # Calculate checksum for data integrity
            data_str = json.dumps(state_data, sort_keys=True, default=str)
            checksum = hashlib.sha256(data_str.encode()).hexdigest()
            state_data['checksum'] = checksum

            # Atomic write using temporary file
            temp_path = path + '.tmp'
            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(state_data, f, indent=2, default=str)

                # Create backup if file exists
                if os.path.exists(path):
                    backup_path = path + '.backup'
                    shutil.copy2(path, backup_path)

                # Atomic move
                os.rename(temp_path, path)

            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise e

    def load(self, path: str) -> bool:
        """Load state from file with validation"""
        if not os.path.exists(path):
            return False

        with self._lock:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)

                # Validate checksum
                if 'checksum' in state_data:
                    data_copy = state_data.copy()
                    expected_checksum = data_copy.pop('checksum')
                    data_str = json.dumps(data_copy, sort_keys=True, default=str)
                    actual_checksum = hashlib.sha256(data_str.encode()).hexdigest()

                    if actual_checksum != expected_checksum:
                        raise ValueError("Data integrity check failed")

                # Load data
                if 'positions' in state_data:
                    self.positions = pd.DataFrame(state_data['positions'])

                if 'orders' in state_data:
                    self.orders = pd.DataFrame(state_data['orders'])

                if 'nav' in state_data:
                    self.nav = state_data['nav']

                return True

            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                # Try loading from backup if available
                backup_path = path + '.backup'
                if os.path.exists(backup_path):
                    try:
                        return self.load(backup_path)
                    except:
                        pass
                return False

    def backup(self, backup_dir: str):
        """Create timestamped backup"""
        with self._lock:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(backup_dir, f'state_backup_{timestamp}.json')

            # Ensure backup directory exists
            os.makedirs(backup_dir, exist_ok=True)

            # Save to backup location
            self.save(backup_path)

    def restore_from_backup(self, backup_path: str) -> bool:
        """Restore state from backup file"""
        return self.load(backup_path)

    def clear(self):
        """Clear all state data"""
        with self._lock:
            self.positions = pd.DataFrame(columns=["code","shares","avg_price","weight"])
            self.orders = pd.DataFrame(columns=["order_id","code","side","shares","status"])
            self.nav = []

    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current state"""
        with self._lock:
            return {
                'position_count': len(self.positions),
                'order_count': len(self.orders),
                'nav_entries': len(self.nav),
                'total_value': self.positions['shares'].sum() if not self.positions.empty else 0,
                'last_update': self.nav[-1]['ts'] if self.nav else None
            }

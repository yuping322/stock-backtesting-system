"""test_state_store_persistence: test cases for StateStore persistence functionality
"""
import pytest
import tempfile
import os
import pandas as pd
from unittest.mock import patch, mock_open
from live_trading.state_store import StateStore


class TestStateStorePersistence:
    """Test cases for StateStore persistence functionality"""

    def test_save_and_load_positions(self):
        """Test saving and loading positions data"""
        store = StateStore()

        # Add some test positions
        positions_data = pd.DataFrame({
            'code': ['600000', '000001', '000002'],
            'shares': [1000, 2000, 1500],
            'avg_price': [10.5, 8.3, 15.2],
            'weight': [0.3, 0.4, 0.3]
        })
        store.positions = positions_data

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'test_positions.json')

            # Test save
            store.save(file_path)

            # Verify file was created
            assert os.path.exists(file_path)

            # Test load
            new_store = StateStore()
            loaded = new_store.load(file_path)
            assert loaded

            # Verify data integrity
            pd.testing.assert_frame_equal(store.positions, new_store.positions)

    def test_save_orders_state(self):
        """Test saving orders state"""
        store = StateStore()

        # Add some test orders
        orders_data = pd.DataFrame({
            'order_id': ['ORD001', 'ORD002'],
            'code': ['600000', '000001'],
            'side': ['buy', 'sell'],
            'shares': [1000, 500],
            'status': ['filled', 'pending']
        })
        store.orders = orders_data

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'test_orders.json')

            # Test save
            store.save(file_path)

            # Verify file was created
            assert os.path.exists(file_path)

            # Test load
            new_store = StateStore()
            loaded = new_store.load(file_path)
            assert loaded

            # Verify data integrity
            pd.testing.assert_frame_equal(store.orders, new_store.orders)

    def test_save_nav_history(self):
        """Test saving NAV history"""
        store = StateStore()

        # Add some NAV entries
        store.nav = [
            {'ts': '20250101', 'nav': 1000000.0, 'pnl': 0.0},
            {'ts': '20250102', 'nav': 1005000.0, 'pnl': 5000.0},
            {'ts': '20250103', 'nav': 995000.0, 'pnl': -5000.0}
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'test_nav.json')

            # Test save
            store.save(file_path)

            # Verify file was created
            assert os.path.exists(file_path)

            # Test load
            new_store = StateStore()
            loaded = new_store.load(file_path)
            assert loaded

            # Verify data integrity
            assert store.nav == new_store.nav

    def test_persistence_error_handling(self):
        """Test error handling during persistence operations"""
        store = StateStore()

        # Test save to invalid path
        with pytest.raises(Exception):
            store.save('/invalid/path/that/does/not/exist/file.json')

    def test_state_backup_and_recovery(self):
        """Test state backup and recovery functionality"""
        store = StateStore()

        # Setup initial state
        store.positions = pd.DataFrame({
            'code': ['600000'],
            'shares': [1000],
            'avg_price': [10.0],
            'weight': [1.0]
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            backup_path = os.path.join(temp_dir, 'backup.json')

            # Test backup
            store.backup(temp_dir)
            backup_files = [f for f in os.listdir(temp_dir) if f.startswith('state_backup_')]
            assert len(backup_files) == 1

            # Modify state
            store.positions = pd.DataFrame({
                'code': ['600000'],
                'shares': [500],  # Changed
                'avg_price': [10.0],
                'weight': [1.0]
            })

            # Test recovery
            backup_file_path = os.path.join(temp_dir, backup_files[0])
            recovered = store.restore_from_backup(backup_file_path)
            assert recovered
            assert store.positions.iloc[0]['shares'] == 1000

    def test_concurrent_access_protection(self):
        """Test protection against concurrent access during persistence"""
        import threading
        import time

        store = StateStore()

        store.positions = pd.DataFrame({
            'code': ['600000'],
            'shares': [1000],
            'avg_price': [10.0],
            'weight': [1.0]
        })

        results = []
        errors = []

        def worker(worker_id):
            try:
                for i in range(10):
                    # Modify state
                    store.positions.loc[0, 'shares'] = 1000 + worker_id * 100 + i
                    # Save state
                    with tempfile.TemporaryDirectory() as temp_dir:
                        file_path = os.path.join(temp_dir, f'test_{worker_id}_{i}.json')
                        store.save(file_path)
                        results.append(f'worker_{worker_id}_save_{i}')
                    time.sleep(0.01)  # Small delay to encourage race conditions
            except Exception as e:
                errors.append(str(e))

        # Start multiple threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Verify no errors occurred (concurrent access should be protected)
        assert len(errors) == 0
        assert len(results) == 30  # 3 workers * 10 saves each

    def test_data_integrity_validation(self):
        """Test data integrity validation during save/load"""
        store = StateStore()

        # Test with valid data
        store.positions = pd.DataFrame({
            'code': ['600000', '000001', '000002'],
            'shares': [1000, 2000, 1500],
            'avg_price': [10.0, 11.0, 12.0],
            'weight': [0.5, 0.3, 0.2]
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'test_integrity.json')

            # Save and load
            store.save(file_path)
            new_store = StateStore()
            loaded = new_store.load(file_path)
            assert loaded

            # Verify data integrity
            pd.testing.assert_frame_equal(store.positions, new_store.positions)

    def test_large_state_persistence_performance(self):
        """Test performance of persisting large state"""
        store = StateStore()

        # Create large positions dataset
        n_positions = 1000
        large_positions = pd.DataFrame({
            'code': [f'{i:06d}' for i in range(n_positions)],
            'shares': [1000] * n_positions,
            'avg_price': [10.0] * n_positions,
            'weight': [1.0/n_positions] * n_positions
        })
        store.positions = large_positions

        # Create large NAV history
        store.nav = [{'ts': f'2025{i:04d}', 'nav': 1000000.0 + i*1000} for i in range(1000)]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'large_state.json')

            import time
            start_time = time.time()

            # Test save performance
            store.save(file_path)

            end_time = time.time()
            duration = end_time - start_time

            # Should complete within reasonable time
            assert duration < 5.0, f"Persistence took too long: {duration}s"
            assert os.path.exists(file_path)

            # Test load performance
            new_store = StateStore()
            start_time = time.time()
            loaded = new_store.load(file_path)
            end_time = time.time()
            load_duration = end_time - start_time

            assert loaded
            assert load_duration < 5.0, f"Loading took too long: {load_duration}s"

    def test_backup_file_creation(self):
        """Test that backup files are created during save"""
        store = StateStore()
        store.positions = pd.DataFrame({
            'code': ['600000'],
            'shares': [1000],
            'avg_price': [10.0],
            'weight': [1.0]
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'test_backup.json')

            # First save
            store.save(file_path)
            assert os.path.exists(file_path)
            assert not os.path.exists(file_path + '.backup')

            # Second save should create backup
            store.positions.loc[0, 'shares'] = 2000
            store.save(file_path)

            assert os.path.exists(file_path + '.backup')

    def test_corrupted_file_recovery(self):
        """Test recovery from corrupted files"""
        store = StateStore()
        store.positions = pd.DataFrame({
            'code': ['600000'],
            'shares': [1000],
            'avg_price': [10.0],
            'weight': [1.0]
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'test_corrupt.json')

            # Save valid file
            store.save(file_path)

            # Corrupt the file with invalid JSON
            with open(file_path, 'w') as f:
                f.write('{invalid json content')

            # Try to load corrupted file
            new_store = StateStore()
            loaded = new_store.load(file_path)
            assert not loaded  # Should fail due to corruption

    def test_state_summary_functionality(self):
        """Test state summary functionality"""
        store = StateStore()

        # Empty state
        summary = store.get_state_summary()
        assert summary['position_count'] == 0
        assert summary['order_count'] == 0
        assert summary['nav_entries'] == 0
        assert summary['total_value'] == 0
        assert summary['last_update'] is None

        # Add data
        store.positions = pd.DataFrame({
            'code': ['600000', '000001'],
            'shares': [1000, 2000],
            'avg_price': [10.0, 8.0],
            'weight': [0.4, 0.6]
        })

        store.orders = pd.DataFrame({
            'order_id': ['ORD001'],
            'code': ['600000'],
            'side': ['buy'],
            'shares': [500],
            'status': ['pending']
        })

        store.nav = [
            {'ts': '20250101', 'nav': 1000000.0},
            {'ts': '20250102', 'nav': 1010000.0}
        ]

        summary = store.get_state_summary()
        assert summary['position_count'] == 2
        assert summary['order_count'] == 1
        assert summary['nav_entries'] == 2
        assert summary['total_value'] == 3000  # 1000 + 2000
        assert summary['last_update'] == '20250102'
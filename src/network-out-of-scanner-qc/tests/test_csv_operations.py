import pandas as pd
import numpy as np
import pytest
import tempfile
import os
from pathlib import Path
from utils.utils import (
    update_qc_csv,
    append_summary_rows_to_csv,
    initialize_qc_csvs,
    get_task_columns
)

class TestCSVOperations:
    """Test CSV operation functions."""
    
    def setup_method(self):
        """Set up test data and temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = Path(self.temp_dir)
        
        # Create sample task data
        self.sample_metrics = {
            'condition_A_acc': 0.8,
            'condition_A_rt': 0.5,
            'condition_B_acc': 0.7,
            'condition_B_rt': 0.6
        }
        
    def teardown_method(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_initialize_qc_csvs(self):
        """Test initialization of QC CSV files."""
        tasks = ['flanker', 'n_back', 'stop_signal']
        
        initialize_qc_csvs(tasks, self.output_path)
        
        # Check that files were created
        for task in tasks:
            csv_file = self.output_path / f"{task}_qc.csv"
            assert csv_file.exists()
            
            # Check that file has correct columns
            df = pd.read_csv(csv_file)
            assert 'subject_id' in df.columns
            
    def test_update_qc_csv_new_file(self):
        """Test updating QC CSV with new subject."""
        task_name = 'flanker'
        subject_id = 's123'
        
        # Update CSV (should create new file)
        update_qc_csv(self.output_path, task_name, subject_id, self.sample_metrics)
        
        # Check that file was created
        csv_file = self.output_path / f"{task_name}_qc.csv"
        assert csv_file.exists()
        
        # Check content
        df = pd.read_csv(csv_file)
        assert len(df) == 1
        assert df.iloc[0]['subject_id'] == subject_id
        assert df.iloc[0]['condition_A_acc'] == 0.8
        assert df.iloc[0]['condition_A_rt'] == 0.5
        
    def test_update_qc_csv_existing_file(self):
        """Test updating existing QC CSV file."""
        task_name = 'flanker'
        
        # Create initial CSV
        initial_df = pd.DataFrame({
            'subject_id': ['s100'],
            'condition_A_acc': [0.9],
            'condition_A_rt': [0.4]
        })
        csv_file = self.output_path / f"{task_name}_qc.csv"
        initial_df.to_csv(csv_file, index=False)
        
        # Add new subject
        update_qc_csv(self.output_path, task_name, 's123', self.sample_metrics)
        
        # Check content
        df = pd.read_csv(csv_file)
        assert len(df) == 2
        assert 's100' in df['subject_id'].values
        assert 's123' in df['subject_id'].values
        
    def test_update_qc_csv_subject_ordering(self):
        """Test that subjects are ordered numerically."""
        task_name = 'flanker'
        
        # Add subjects in non-numerical order
        update_qc_csv(self.output_path, task_name, 's1175', {'condition_A_acc': 0.8})
        update_qc_csv(self.output_path, task_name, 's956', {'condition_A_acc': 0.7})
        update_qc_csv(self.output_path, task_name, 's123', {'condition_A_acc': 0.9})
        
        # Check ordering
        df = pd.read_csv(self.output_path / f"{task_name}_qc.csv")
        expected_order = ['s123', 's956', 's1175']
        assert list(df['subject_id']) == expected_order
        
    def test_update_qc_csv_new_columns(self):
        """Test adding new columns to existing CSV."""
        task_name = 'flanker'
        
        # Create initial CSV with some columns
        initial_df = pd.DataFrame({
            'subject_id': ['s100'],
            'condition_A_acc': [0.9]
        })
        csv_file = self.output_path / f"{task_name}_qc.csv"
        initial_df.to_csv(csv_file, index=False)
        
        # Add new subject with additional columns
        new_metrics = {
            'condition_A_acc': 0.8,
            'condition_B_acc': 0.7,  # New column
            'condition_B_rt': 0.6    # New column
        }
        update_qc_csv(self.output_path, task_name, 's123', new_metrics)
        
        # Check that new columns were added
        df = pd.read_csv(csv_file)
        assert 'condition_B_acc' in df.columns
        assert 'condition_B_rt' in df.columns
        
        # Check that original subject has NaN for new columns
        assert pd.isna(df.loc[df['subject_id'] == 's100', 'condition_B_acc'].iloc[0])
        
    def test_update_qc_csv_data_type_handling(self):
        """Test handling of different data types."""
        task_name = 'flanker'
        
        # Test with mixed data types
        mixed_metrics = {
            'numeric_col': 0.8,
            'string_col': 'test_value',
            'int_col': 42
        }
        
        update_qc_csv(self.output_path, task_name, 's123', mixed_metrics)
        
        # Check data types
        df = pd.read_csv(self.output_path / f"{task_name}_qc.csv")
        assert pd.api.types.is_numeric_dtype(df['numeric_col'])
        assert pd.api.types.is_string_dtype(df['string_col'])
        assert pd.api.types.is_numeric_dtype(df['int_col'])
        
    def test_append_summary_rows_to_csv(self):
        """Test appending summary statistics to CSV."""
        # Create sample CSV
        csv_file = self.output_path / "test_qc.csv"
        df = pd.DataFrame({
            'subject_id': ['s100', 's101', 's102'],
            'condition_A_acc': [0.8, 0.9, 0.7],
            'condition_A_rt': [0.5, 0.4, 0.6],
            'condition_B_acc': [0.6, 0.8, 0.7],
            'condition_B_rt': [0.7, 0.5, 0.6]
        })
        df.to_csv(csv_file, index=False)
        
        # Append summary rows
        append_summary_rows_to_csv(csv_file)
        
        # Check that summary rows were added
        df_updated = pd.read_csv(csv_file)
        assert len(df_updated) == 7  # 3 original + 4 summary rows
        
        # Check summary row values
        summary_rows = df_updated.tail(4)
        assert 'mean' in summary_rows['subject_id'].values
        assert 'std' in summary_rows['subject_id'].values
        assert 'max' in summary_rows['subject_id'].values
        assert 'min' in summary_rows['subject_id'].values
        
        # Check that numeric columns have correct summary values
        mean_row = summary_rows[summary_rows['subject_id'] == 'mean'].iloc[0]
        assert mean_row['condition_A_acc'] == pytest.approx(0.8)  # (0.8 + 0.9 + 0.7) / 3
        assert mean_row['condition_A_rt'] == pytest.approx(0.5)   # (0.5 + 0.4 + 0.6) / 3
        
    def test_append_summary_rows_to_empty_csv(self):
        """Test appending summary rows to empty CSV."""
        # Create empty CSV
        csv_file = self.output_path / "empty_qc.csv"
        empty_df = pd.DataFrame(columns=['subject_id', 'condition_A_acc'])
        empty_df.to_csv(csv_file, index=False)
        
        # Should handle gracefully
        append_summary_rows_to_csv(csv_file)
        
        # File should remain unchanged
        df = pd.read_csv(csv_file)
        assert len(df) == 0
        
    def test_append_summary_rows_to_single_column_csv(self):
        """Test appending summary rows to CSV with only subject_id."""
        # Create CSV with only subject_id
        csv_file = self.output_path / "single_col_qc.csv"
        df = pd.DataFrame({
            'subject_id': ['s100', 's101']
        })
        df.to_csv(csv_file, index=False)
        
        # Should handle gracefully
        append_summary_rows_to_csv(csv_file)
        
        # File should remain unchanged
        df_updated = pd.read_csv(csv_file)
        assert len(df_updated) == 2
        
    def test_get_task_columns(self):
        """Test getting columns for different tasks."""
        # Test single tasks
        flanker_columns = get_task_columns('flanker')
        assert 'subject_id' in flanker_columns
        assert 'congruent_acc' in flanker_columns
        assert 'incongruent_acc' in flanker_columns
        
        n_back_columns = get_task_columns('n_back')
        assert 'subject_id' in n_back_columns
        
        stop_signal_columns = get_task_columns('stop_signal')
        assert 'subject_id' in stop_signal_columns
        assert 'go_rt' in stop_signal_columns
        assert 'ssrt' in stop_signal_columns
        
        # Test dual tasks
        dual_columns = get_task_columns('flanker_with_n_back')
        assert 'subject_id' in dual_columns
        
    def test_get_task_columns_with_sample_data(self):
        """Test getting columns with sample data for dynamic tasks."""
        # Create sample data for n-back
        sample_df = pd.DataFrame({
            'n_back_condition': ['0', '2'],
            'delay': [1.0, 2.0],
            'correct_trial': [1, 1],
            'rt': [0.5, 0.7],
            'key_press': [1, 1]
        })
        
        n_back_columns = get_task_columns('n_back', sample_df)
        assert 'subject_id' in n_back_columns
        assert '0_1.0back_acc' in n_back_columns
        assert '2_2.0back_acc' in n_back_columns
        
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with non-existent task
        unknown_columns = get_task_columns('unknown_task')
        assert unknown_columns is None
        
        # Test with empty metrics
        update_qc_csv(self.output_path, 'flanker', 's123', {})
        
        csv_file = self.output_path / "flanker_qc.csv"
        assert csv_file.exists()
        
        df = pd.read_csv(csv_file)
        assert len(df) == 1
        assert df.iloc[0]['subject_id'] == 's123'
        
        # Test with NaN values in metrics
        nan_metrics = {
            'condition_A_acc': np.nan,
            'condition_A_rt': np.nan
        }
        update_qc_csv(self.output_path, 'flanker', 's124', nan_metrics)
        
        df = pd.read_csv(csv_file)
        assert len(df) == 2
        assert pd.isna(df.loc[df['subject_id'] == 's124', 'condition_A_acc'].iloc[0])
        
    def test_file_not_found_handling(self):
        """Test handling of file not found errors."""
        # Test updating non-existent file (should create new one)
        update_qc_csv(self.output_path, 'new_task', 's123', self.sample_metrics)
        
        csv_file = self.output_path / "new_task_qc.csv"
        assert csv_file.exists()
        
    def test_csv_encoding_and_formatting(self):
        """Test CSV encoding and formatting."""
        task_name = 'flanker'
        
        # Add data with special characters
        special_metrics = {
            'condition_with_underscore_acc': 0.8,
            'condition_with.dots_rt': 0.5
        }
        
        update_qc_csv(self.output_path, task_name, 's123', special_metrics)
        
        # Check that file can be read back correctly
        df = pd.read_csv(self.output_path / f"{task_name}_qc.csv")
        assert 'condition_with_underscore_acc' in df.columns
        assert 'condition_with.dots_rt' in df.columns
        assert df.iloc[0]['condition_with_underscore_acc'] == 0.8

if __name__ == "__main__":
    pytest.main([__file__]) 
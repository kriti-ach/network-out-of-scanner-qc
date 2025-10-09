import pandas as pd
import numpy as np
import pytest
import sys
import os

from utils.qc_utils import (
    calculate_acc, calculate_rt, calculate_omission_rate,
    calculate_commission_rate, calculate_basic_metrics
)

class TestBasicMetrics:
    """Test basic metric calculation functions."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample DataFrame with various trial types
        self.df = pd.DataFrame({
            'correct_trial': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
            'rt': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
            'key_press': [1, 1, 2, 1, 2, 1, 2, 1, 1, 2],  # -1 would be omission
            'condition': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']
        })
        
        # Create mask for condition A
        self.mask_a = self.df['condition'] == 'A'
        self.mask_b = self.df['condition'] == 'B'
        
        # Create mask for correct trials only
        self.mask_correct = self.df['correct_trial'] == 1
        
    def test_calculate_acc(self):
        """Test accuracy calculation."""
        # Test condition A (3 correct out of 5 trials)
        acc_a = calculate_acc(self.df, self.mask_a)
        assert acc_a == 0.6
        
        # Test condition B (3 correct out of 5 trials)
        acc_b = calculate_acc(self.df, self.mask_b)
        assert acc_b == 0.6
        
        # Test empty mask
        empty_mask = self.df['condition'] == 'C'
        acc_empty = calculate_acc(self.df, empty_mask)
        assert np.isnan(acc_empty)
        
    def test_calculate_rt(self):
        """Test reaction time calculation."""
        # Test RT for correct trials in condition A
        mask_rt_a = self.mask_a & self.mask_correct
        rt_a = calculate_rt(self.df, mask_rt_a)
        expected_rt_a = np.mean([0.5, 0.6, 0.8])  # Correct trials in A
        assert rt_a == pytest.approx(expected_rt_a)
        
        # Test RT for correct trials in condition B
        mask_rt_b = self.mask_b & self.mask_correct
        rt_b = calculate_rt(self.df, mask_rt_b)
        expected_rt_b = np.mean([1.0, 1.2, 1.3])  # Correct trials in B
        assert rt_b == pytest.approx(expected_rt_b)
        
        # Test empty mask
        empty_mask = self.df['condition'] == 'C'
        rt_empty = calculate_rt(self.df, empty_mask)
        assert np.isnan(rt_empty)
        
    def test_calculate_omission_rate(self):
        """Test omission rate calculation."""
        # Create data with omissions (key_press = -1)
        df_with_omissions = pd.DataFrame({
            'correct_trial': [1, 0, 1, 0, 1],
            'key_press': [1, -1, 1, -1, 1],  # 2 omissions
            'condition': ['A', 'A', 'A', 'A', 'A']
        })
        mask_a = df_with_omissions['key_press'] == -1
        
        omission_rate = calculate_omission_rate(df_with_omissions, mask_a, total_num_trials=5)
        assert omission_rate == 0.4  # 2 omissions / 5 trials
        
        # Test with no omissions
        df_no_omissions = pd.DataFrame({
            'correct_trial': [1, 1, 1],
            'key_press': [1, 1, 1],
            'condition': ['A', 'A', 'A']
        })
        mask_a = df_no_omissions['key_press'] == -1
        
        omission_rate = calculate_omission_rate(df_no_omissions, mask_a, total_num_trials=3)
        assert omission_rate == 0.0
        
    def test_calculate_commission_rate(self):
        """Test commission rate calculation."""
        # Create data with commissions (incorrect responses)
        df_with_commissions = pd.DataFrame({
            'correct_trial': [1, 0, 1, 0, 1],
            'key_press': [1, 2, 1, 2, 1],  # 2 commissions (incorrect responses)
            'condition': ['A', 'A', 'A', 'A', 'A']
        })
        mask_a = df_with_commissions['correct_trial'] == 0
        
        commission_rate = calculate_commission_rate(df_with_commissions, mask_a, total_num_trials=5)
        assert commission_rate == 0.4  # 2 commissions / 5 trials
        
        # Test with no commissions
        df_no_commissions = pd.DataFrame({
            'correct_trial': [1, 1, 1],
            'key_press': [1, 1, 1],
            'condition': ['A', 'A', 'A']
        })
        mask_a = df_no_commissions['correct_trial'] == 0
        
        commission_rate = calculate_commission_rate(df_no_commissions, mask_a, total_num_trials=3)
        assert commission_rate == 0.0
        
    def test_calculate_basic_metrics(self):
        """Test the comprehensive basic metrics function."""
        # Create test data
        df = pd.DataFrame({
            'correct_trial': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
            'rt': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
            'key_press': [1, 1, 2, 1, 2, 1, 2, 1, 1, 2],
            'condition': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']
        })
        
        mask_a = df['condition'] == 'A'
        metrics = {}
        
        calculate_basic_metrics(df, mask_a, 'condition_A', metrics)
        
        # Check that all metrics were calculated
        assert 'condition_A_acc' in metrics
        assert 'condition_A_rt' in metrics
        assert 'condition_A_omission_rate' in metrics
        assert 'condition_A_commission_rate' in metrics
        
        # Check accuracy (3 correct out of 5 trials)
        assert metrics['condition_A_acc'] == 0.6
        
        # Check RT (mean of correct trials: 0.5, 0.6, 0.8)
        expected_rt = np.mean([0.5, 0.6, 0.8])
        assert metrics['condition_A_rt'] == pytest.approx(expected_rt)
        
        # Check omission rate (0 omissions)
        assert metrics['condition_A_omission_rate'] == 0.0
        
        # Check commission rate (2 commissions out of 5 trials)
        assert metrics['condition_A_commission_rate'] == 0.4
        
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame(columns=['correct_trial', 'rt', 'key_press'])
        empty_mask = pd.Series([False] * 0)
        
        assert np.isnan(calculate_acc(empty_df, empty_mask))
        assert np.isnan(calculate_rt(empty_df, empty_mask))
        
        # Test with all NaN values
        nan_df = pd.DataFrame({
            'correct_trial': [np.nan, np.nan, np.nan],
            'rt': [np.nan, np.nan, np.nan],
            'key_press': [np.nan, np.nan, np.nan]
        })
        nan_mask = pd.Series([True, True, True])
        
        assert np.isnan(calculate_acc(nan_df, nan_mask))
        assert np.isnan(calculate_rt(nan_df, nan_mask))
        
        # Test with mixed data types
        mixed_df = pd.DataFrame({
            'correct_trial': [1, 0, 1],
            'rt': [0.5, 0.6, 0.7],
            'key_press': [1, -1, 2]  # Mix of correct, omission, commission
        })
        mask_accuracy = mixed_df['correct_trial'] >= 0
        mask_rt = mixed_df['rt'] > 0
        mask_omission = mixed_df['key_press'] == -1
        mask_commission = mixed_df['correct_trial'] == 0
        
        acc = calculate_acc(mixed_df, mask_accuracy)
        rt = calculate_rt(mixed_df, mask_rt)
        omission = calculate_omission_rate(mixed_df, mask_omission, 3)
        commission = calculate_commission_rate(mixed_df, mask_commission, 3)
        
        assert acc == 2/3  # 2 correct out of 3
        assert rt == pytest.approx(0.6)  # Mean of correct trials (0.5, 0.7)
        assert omission == 1/3  # 1 omission out of 3
        assert commission == 1/3  # 1 commission out of 3

if __name__ == "__main__":
    pytest.main([__file__]) 
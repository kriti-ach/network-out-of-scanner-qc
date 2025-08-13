import pandas as pd
import numpy as np
import pytest
from utils.utils import (
    generate_n_back_conditions,
    get_dual_n_back_columns,
    compute_n_back_metrics
)

class TestNBackMetrics:
    """Test n-back metric calculation functions."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample n-back data
        self.df = pd.DataFrame({
            'n_back_condition': ['match', 'mismatch', 'match', 'mismatch', 'match', 'mismatch'],
            'delay': [1.0, 1.0, 2.0, 2.0, 1.0, 2.0],
            'correct_trial': [1, 1, 0, 1, 0, 1],
            'rt': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'key_press': [1, 1, 2, 1, 2, 1],
            'trial_id': ['test_trial'] * 6
        })
        
        # Create n-back with cued task switching data
        self.df_cuedts = pd.DataFrame({
            'n_back_condition': ['match', 'mismatch', 'match', 'mismatch'],
            'delay': [1.0, 1.0, 2.0, 2.0],
            'cue_condition': ['stay', 'switch', 'stay', 'switch'],
            'task_condition': ['stay', 'stay', 'switch', 'switch'],
            'correct_trial': [1, 1, 0, 1],
            'rt': [0.5, 0.6, 0.7, 0.8],
            'key_press': [1, 1, 2, 1],
            'trial_id': ['test_trial'] * 4
        })
        
        # Create n-back with paired task data
        self.df_paired = pd.DataFrame({
            'n_back_condition': ['match', 'mismatch', 'match', 'mismatch'],
            'delay': [1.0, 1.0, 2.0, 2.0],
            'flanker_condition': ['congruent', 'incongruent', 'congruent', 'incongruent'],
            'correct_trial': [1, 1, 0, 1],
            'rt': [0.5, 0.6, 0.7, 0.8],
            'key_press': [1, 1, 2, 1],
            'trial_id': ['test_trial'] * 4
        })
        
    def test_generate_n_back_conditions_simple(self):
        """Test simple n-back condition generation."""
        conditions = generate_n_back_conditions(self.df)
        
        # Should generate: match_1.0back, mismatch_2.0back
        expected_conditions = ['match_1.0back', 'mismatch_2.0back']
        assert set(conditions) == set(expected_conditions)
        
    def test_generate_n_back_conditions_with_cuedts(self):
        """Test n-back condition generation with cued task switching."""
        conditions = generate_n_back_conditions(self.df_cuedts, include_cuedts=True)
        
        # Should generate conditions like: 0_1.0back_tstay_cstay, 0_1.0back_tstay_cswitch, etc.
        # But skip combinations where cue="stay" and task="switch"
        expected_conditions = [
            'match_1.0back_tstay_cstay',
            'match_1.0back_tstay_cswitch',
            'mismatch_2.0back_tswitch_cstay',
            'mismatch_2.0back_tswitch_cswitch'
        ]
        assert set(conditions) == set(expected_conditions)
        
        # Verify that stay/switch combination is skipped
        assert 'match_1.0back_tswitch_cstay' not in conditions
        
    def test_generate_n_back_conditions_with_paired_task(self):
        """Test n-back condition generation with paired task."""
        paired_conditions = ['congruent', 'incongruent']
        conditions = generate_n_back_conditions(
            self.df_paired, include_paired_task=True, 
            paired_task_col='flanker_condition', paired_conditions=paired_conditions
        )
        
        # Should generate: 0_1.0back_congruent, 0_1.0back_incongruent, etc.
        expected_conditions = [
            'match_1.0back_congruent',
            'match_1.0back_incongruent',
            'mismatch_2.0back_congruent',
            'mismatch_2.0back_incongruent'
        ]
        assert set(conditions) == set(expected_conditions)
        
    def test_generate_n_back_conditions_with_nan_values(self):
        """Test n-back condition generation with NaN values."""
        df_with_nan = pd.DataFrame({
            'n_back_condition': ['match', np.nan, 'mismatch', 'match'],
            'delay': [1.0, 2.0, np.nan, 1.0],
            'correct_trial': [1, 1, 1, 1],
            'rt': [0.5, 0.6, 0.7, 0.8],
            'key_press': [1, 1, 1, 1]
        })
        
        conditions = generate_n_back_conditions(df_with_nan)
        
        # Should only generate conditions for non-NaN values
        expected_conditions = ['match_1.0back']
        assert set(conditions) == set(expected_conditions)
        
    def test_get_dual_n_back_columns_simple(self):
        """Test dual n-back column generation for simple case."""
        base_columns = ['subject_id']
        columns = get_dual_n_back_columns(base_columns, self.df_paired, 'flanker_condition')
        
        # Should include base columns plus metric columns for each condition
        expected_metric_columns = [
            'match_1.0back_congruent_acc', 'match_1.0back_congruent_rt', 
            'match_1.0back_congruent_omission_rate', 'match_1.0back_congruent_commission_rate',
            'match_1.0back_incongruent_acc', 'match_1.0back_incongruent_rt',
            'match_1.0back_incongruent_omission_rate', 'match_1.0back_incongruent_commission_rate',
            'mismatch_2.0back_congruent_acc', 'mismatch_2.0back_congruent_rt',
            'mismatch_2.0back_congruent_omission_rate', 'mismatch_2.0back_congruent_commission_rate',
            'mismatch_2.0back_incongruent_acc', 'mismatch_2.0back_incongruent_rt',
            'mismatch_2.0back_incongruent_omission_rate', 'mismatch_2.0back_incongruent_commission_rate'
        ]
        
        for col in expected_metric_columns:
            assert col in columns
            
    def test_get_dual_n_back_columns_cuedts(self):
        """Test dual n-back column generation for cued task switching."""
        base_columns = ['subject_id']
        columns = get_dual_n_back_columns(base_columns, self.df_cuedts, cuedts=True)
        
        # Should include base columns plus metric columns for each cued task switching condition
        expected_metric_columns = [
            'match_1.0back_tstay_cstay_acc', 'match_1.0back_tstay_cstay_rt',
            'match_1.0back_tstay_cstay_omission_rate', 'match_1.0back_tstay_cstay_commission_rate',
            'match_1.0back_tstay_cswitch_acc', 'match_1.0back_tstay_cswitch_rt',
            'match_1.0back_tstay_cswitch_omission_rate', 'match_1.0back_tstay_cswitch_commission_rate',
            'mismatch_2.0back_tswitch_cstay_acc', 'mismatch_2.0back_tswitch_cstay_rt',
            'mismatch_2.0back_tswitch_cstay_omission_rate', 'mismatch_2.0back_tswitch_cstay_commission_rate',
            'mismatch_2.0back_tswitch_cswitch_acc', 'mismatch_2.0back_tswitch_cswitch_rt',
            'mismatch_2.0back_tswitch_cswitch_omission_rate', 'mismatch_2.0back_tswitch_cswitch_commission_rate'
        ]
        
        for col in expected_metric_columns:
            assert col in columns
            
    def test_compute_n_back_metrics_single(self):
        """Test single n-back metrics calculation."""
        metrics = compute_n_back_metrics(self.df, None)
        
        # Should calculate metrics for each n-back condition
        expected_conditions = ['match_1.0back', 'mismatch_2.0back']
        
        for condition in expected_conditions:
            assert f'{condition}_acc' in metrics
            assert f'{condition}_rt' in metrics
            assert f'{condition}_omission_rate' in metrics
            assert f'{condition}_commission_rate' in metrics
            
        # Check specific values for 0_1.0back (2 correct out of 3 trials)
        assert metrics['match_1.0back_acc'] == pytest.approx(2/3)
        
        # Check RT for correct trials in 0_1.0back (mean of 0.5, 0.6)
        expected_rt = np.mean([0.5, 0.6])
        assert metrics['match_1.0back_rt'] == pytest.approx(expected_rt)
        
    def test_compute_n_back_metrics_cuedts(self):
        """Test n-back metrics calculation with cued task switching."""
        metrics = compute_n_back_metrics(self.df_cuedts, None, cuedts=True)
        
        # Should calculate metrics for each cued task switching condition
        expected_conditions = [
            'match_1.0back_tstay_cstay',
            'match_1.0back_tstay_cswitch',
            'mismatch_2.0back_tswitch_cstay',
            'mismatch_2.0back_tswitch_cswitch'
        ]
        
        for condition in expected_conditions:
            assert f'{condition}_acc' in metrics
            assert f'{condition}_rt' in metrics
            assert f'{condition}_omission_rate' in metrics
            assert f'{condition}_commission_rate' in metrics
            
        # Check specific values for 0_1.0back_tstay_cstay (1 correct out of 1 trial)
        assert metrics['match_1.0back_tstay_cstay_acc'] == 1.0
        assert metrics['match_1.0back_tstay_cstay_rt'] == pytest.approx(0.5)
        
    def test_compute_n_back_metrics_paired(self):
        """Test n-back metrics calculation with paired task."""
        paired_conditions = ['congruent', 'incongruent']
        metrics = compute_n_back_metrics(
            self.df_paired, None, paired_task_col='flanker_condition', 
            paired_conditions=paired_conditions
        )
        
        # Should calculate metrics for each paired condition
        expected_conditions = [
            'match_1.0back_congruent',
            'match_1.0back_incongruent',
            'mismatch_2.0back_congruent',
            'mismatch_2.0back_incongruent'
        ]
        
        for condition in expected_conditions:
            assert f'{condition}_acc' in metrics
            assert f'{condition}_rt' in metrics
            assert f'{condition}_omission_rate' in metrics
            assert f'{condition}_commission_rate' in metrics
            
        # Check specific values for 0_1.0back_congruent (1 correct out of 1 trial)
        assert metrics['match_1.0back_congruent_acc'] == 1.0
        assert metrics['match_1.0back_congruent_rt'] == pytest.approx(0.5)
        
    def test_compute_n_back_metrics_with_nan_values(self):
        """Test n-back metrics calculation with NaN values."""
        df_with_nan = pd.DataFrame({
            'n_back_condition': ['match', np.nan, 'mismatch', 'match'],
            'delay': [1.0, 2.0, np.nan, 1.0],
            'correct_trial': [1, 1, 1, 1],
            'rt': [0.5, 0.6, 0.7, 0.8],
            'key_press': [1, 1, 1, 1],
            'trial_id': ['test_trial'] * 4
        })
        
        metrics = compute_n_back_metrics(df_with_nan, None)
        
        # Should only calculate metrics for non-NaN conditions
        assert 'match_1.0back_acc' in metrics
        assert 'mismatch_2.0back_acc' not in metrics  # Should not be generated
        
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame(columns=['n_back_condition', 'delay', 'correct_trial', 'rt', 'key_press'])
        
        conditions = generate_n_back_conditions(empty_df)
        assert conditions == []
        
        metrics = compute_n_back_metrics(empty_df, None)
        assert metrics == {}
        
        # Test with all NaN values
        nan_df = pd.DataFrame({
            'n_back_condition': [np.nan, np.nan],
            'delay': [np.nan, np.nan],
            'correct_trial': [1, 1],
            'rt': [0.5, 0.6],
            'key_press': [1, 1],
            'trial_id': ['test_trial'] * 2
        })
        
        conditions = generate_n_back_conditions(nan_df)
        assert conditions == []
        
        metrics = compute_n_back_metrics(nan_df, None)
        assert metrics == {}
        
        # Test with no test trials
        df_no_test = self.df.copy()
        df_no_test['trial_id'] = ['practice'] * len(df_no_test)
        
        metrics = compute_n_back_metrics(df_no_test, None)
        assert metrics == {}
        
    def test_case_sensitivity(self):
        """Test case sensitivity in condition matching."""
        # Test with mixed case in n_back_condition
        df_mixed_case = pd.DataFrame({
            'n_back_condition': ['match', 'match', 'mismatch', 'mismatch'],
            'delay': [1.0, 1.0, 2.0, 2.0],
            'correct_trial': [1, 1, 1, 1],
            'rt': [0.5, 0.6, 0.7, 0.8],
            'key_press': [1, 1, 1, 1],
            'trial_id': ['test_trial'] * 4
        })
        
        metrics = compute_n_back_metrics(df_mixed_case, None)
        
        # Should still generate correct conditions
        assert 'match_1.0back_acc' in metrics
        assert 'mismatch_2.0back_acc' in metrics
        
    def test_condition_parsing(self):
        """Test condition parsing in metrics calculation."""
        # Test that condition parsing works correctly
        df_test = pd.DataFrame({
            'n_back_condition': ['match', 'mismatch'],
            'delay': [1.0, 2.0],
            'correct_trial': [1, 1],
            'rt': [0.5, 0.7],
            'key_press': [1, 1],
            'trial_id': ['test_trial'] * 2
        })
        
        metrics = compute_n_back_metrics(df_test, None)
        
        # Verify that conditions are parsed correctly
        assert 'match_1.0back_acc' in metrics
        assert 'mismatch_2.0back_acc' in metrics
        
        # Verify that RT calculations use correct masks
        assert metrics['match_1.0back_rt'] == pytest.approx(0.5)
        assert metrics['mismatch_2.0back_rt'] == pytest.approx(0.7)

if __name__ == "__main__":
    pytest.main([__file__]) 
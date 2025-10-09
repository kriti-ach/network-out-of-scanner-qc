import pandas as pd
import numpy as np
import pytest
from utils.qc_utils import (
    calculate_single_stop_signal_metrics,
    calculate_stop_signal_ssd_stats,
    calculate_dual_stop_signal_condition_metrics,
    parse_dual_task_condition,
    compute_stop_signal_metrics,
    get_go_trials_rt,
    get_stop_trials_info,
    get_nth_rt,
    compute_SSRT
)

class TestStopSignalMetrics:
    """Test stop signal metric calculation functions."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample stop signal data
        self.df = pd.DataFrame({
            'SS_trial_type': ['go', 'go', 'stop', 'go', 'stop', 'go', 'stop', 'go'],
            'correct_trial': [1, 1, 0, 1, 0, 1, 1, 1],  # Stop failures are incorrect
            'rt': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, np.nan, 1.2],  # Stop success has no RT
            'key_press': [1, 1, 2, 1, 2, 1, -1, 1],  # -1 for stop success
            'SS_delay': [np.nan, np.nan, 0.2, np.nan, 0.3, np.nan, 0.4, np.nan],
            'stim': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            'correct_response': [1, 2, 1, 2, 1, 2, 1, 2]
        })
        
    def test_calculate_single_stop_signal_metrics(self):
        """Test single stop signal metrics calculation."""
        metrics = calculate_single_stop_signal_metrics(self.df)
        
        # Check that all expected metrics are present
        expected_keys = ['go_rt', 'stop_fail_rt', 'go_acc', 'stop_fail_acc', 'stop_success']
        for key in expected_keys:
            assert key in metrics
        
        # Check go RT (mean of go trial RTs: 0.5, 0.6, 0.8, 1.0, 1.2)
        expected_go_rt = np.mean([0.5, 0.6, 0.8, 1.0, 1.2])
        assert metrics['go_rt'] == pytest.approx(expected_go_rt)
        
        # Check stop failure RT (mean of stop failure RTs: 0.7, 0.9)
        expected_stop_fail_rt = np.mean([0.7, 0.9])
        assert metrics['stop_fail_rt'] == pytest.approx(expected_stop_fail_rt)
        
        # Check go accuracy (all go trials correct: 5/5)
        assert metrics['go_acc'] == 1.0
        
        # Check stop success rate (1 success out of 3 stop trials)
        assert metrics['stop_success'] == pytest.approx(1/3)
        
    def test_calculate_stop_signal_ssd_stats(self):
        """Test SSD statistics calculation."""
        metrics = calculate_stop_signal_ssd_stats(self.df)
        
        # Check that all expected metrics are present
        expected_keys = ['avg_ssd', 'min_ssd', 'max_ssd', 'min_ssd_count', 'max_ssd_count']
        for key in expected_keys:
            assert key in metrics
        
        # Check SSD values (0.2, 0.3, 0.4)
        assert metrics['avg_ssd'] == pytest.approx(0.3)
        assert metrics['min_ssd'] == pytest.approx(0.2)
        assert metrics['max_ssd'] == pytest.approx(0.4)
        assert metrics['min_ssd_count'] == 1
        assert metrics['max_ssd_count'] == 1
        
    def test_calculate_dual_stop_signal_condition_metrics(self):
        """Test dual stop signal condition metrics."""
        # Create test data with paired task condition
        df_dual = pd.DataFrame({
            'SS_trial_type': ['go', 'go', 'stop', 'go', 'stop'],
            'correct_trial': [1, 1, 0, 1, 0],
            'rt': [0.5, 0.6, 0.7, 0.8, 0.9],
            'key_press': [1, 1, 2, 1, 2],
            'flanker_condition': ['congruent', 'congruent', 'congruent', 'incongruent', 'incongruent'],
            'SS_delay': [np.nan, np.nan, 0.2, np.nan, 0.3],
            'stim': ['A', 'B', 'A', 'B', 'A'],
            'correct_response': [1, 2, 1, 2, 1]
        })
        
        # Test with single stimulus column
        paired_mask = df_dual['flanker_condition'] == 'congruent'
        metrics = calculate_dual_stop_signal_condition_metrics(
            df_dual, 'congruent', paired_mask, stim_col='stim'
        )
        
        # Check that metrics are calculated for the condition
        assert 'congruent_go_rt' in metrics
        assert 'congruent_stop_fail_rt' in metrics
        assert 'congruent_go_acc' in metrics
        assert 'congruent_stop_fail_acc' in metrics
        assert 'congruent_stop_success' in metrics
        assert 'congruent_ssrt' in metrics  # Now includes SSRT
        
        # Check values
        assert metrics['congruent_go_rt'] == pytest.approx(np.mean([0.5, 0.6]))
        # In this dataset, one go trial under congruent has correct_response 2 but key_press 1,
        # so go_acc should be 0.5
        assert metrics['congruent_go_acc'] == 0.5
        assert metrics['congruent_stop_success'] == 0  
        
    def test_parse_dual_task_condition(self):
        """Test dual task condition parsing."""
        # Test simple condition
        mask_func, args = parse_dual_task_condition('congruent', 'flanker_condition')
        assert mask_func is not None
        assert args is None
        
        # Test n-back condition
        mask_func, args = parse_dual_task_condition('0_1back', None)
        assert mask_func is not None
        assert args is None
        
        # Test cued task switching condition
        mask_func, args = parse_dual_task_condition('tstay_cstay', None)
        assert mask_func is not None
        assert args is None
        
        # Test invalid condition
        mask_func, args = parse_dual_task_condition('invalid', None)
        assert mask_func is None
        assert args is None
        
    def test_get_go_trials_rt(self):
        """Test go trial RT extraction."""
        # Test without condition mask (original behavior)
        sorted_rt = get_go_trials_rt(self.df)
        
        # The function preserves original indices, so we need to match that
        # Go trials are at indices 0, 1, 3, 5, 7 with RTs [0.5, 0.6, 0.8, 1.0, 1.2]
        expected_rt = pd.Series([0.5, 0.6, 0.8, 1.0, 1.2], index=[0, 1, 3, 5, 7], name='rt')
        pd.testing.assert_series_equal(sorted_rt, expected_rt) 
        
        # Test with condition mask
        condition_mask = self.df['SS_trial_type'] == 'go'
        sorted_rt = get_go_trials_rt(self.df, condition_mask=condition_mask)
        pd.testing.assert_series_equal(sorted_rt, expected_rt)
        
        # Test with missing RT values (should be filled with max_go_rt)
        df_missing = self.df.copy()
        df_missing.loc[0, 'rt'] = np.nan
        sorted_rt = get_go_trials_rt(df_missing, max_go_rt=2.0)
        pd.testing.assert_series_equal(
            sorted_rt,
            pd.Series([0.6, 0.8, 1.0, 1.2, 2.0], index=[1, 3, 5, 7, 0], name='rt')
        )
        
    def test_get_stop_trials_info(self):
        """Test stop trial information extraction."""
        # Test without condition mask (original behavior)
        p_respond, avg_ssd = get_stop_trials_info(self.df)
        
        # Check probability of responding (2 failures out of 3 stop trials)
        assert p_respond == pytest.approx(2/3)
        
        # Check average SSD
        assert avg_ssd == pytest.approx(0.3)
        
        # Test with condition mask
        condition_mask = self.df['SS_trial_type'] == 'stop'
        p_respond, avg_ssd = get_stop_trials_info(self.df, condition_mask=condition_mask)
        assert p_respond == pytest.approx(2/3)
        assert avg_ssd == pytest.approx(0.3)
        
        # Test with no stop trials
        df_no_stop = self.df[self.df['SS_trial_type'] == 'go']
        p_respond, avg_ssd = get_stop_trials_info(df_no_stop)
        assert p_respond == 0.0
        assert np.isnan(avg_ssd)
        
    def test_get_nth_rt(self):
        """Test nth RT calculation."""
        sorted_rt = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        
        # Test middle value
        nth_rt = get_nth_rt(sorted_rt, 0.5)
        assert nth_rt == 0.2
        
        # Test edge cases
        nth_rt = get_nth_rt(sorted_rt, 0.0)
        assert nth_rt == 0.1
        
        nth_rt = get_nth_rt(sorted_rt, 1.0)
        assert nth_rt == 0.5
        
        # Test empty series
        nth_rt = get_nth_rt(pd.Series([]), 0.5)
        assert np.isnan(nth_rt)
        
    def test_compute_SSRT(self):
        """Test SSRT calculation."""
        # Test without condition mask (original behavior)
        ssrt = compute_SSRT(self.df)
        expected_ssrt = 0.5
        assert ssrt == expected_ssrt
        
        # Test with no go trials
        df_no_go = self.df[self.df['SS_trial_type'] == 'stop']
        ssrt = compute_SSRT(df_no_go)
        assert np.isnan(ssrt)
        
    def test_compute_SSRT_with_condition_mask(self):
        """Test SSRT calculation with condition masking for dual tasks."""
        # Create dual task data with different conditions
        df_dual = pd.DataFrame({
            'SS_trial_type': ['go', 'go', 'stop', 'go', 'stop', 'go', 'stop'],
            'rt': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
            'key_press': [1, 1, 2, 1, 2, 1, -1],
            'SS_delay': [np.nan, np.nan, 0.2, np.nan, 0.3, np.nan, 0.4],
            'flanker_condition': ['congruent', 'congruent', 'congruent', 'incongruent', 'incongruent', 'incongruent', 'incongruent']
        })
        
        # Test SSRT for congruent condition only
        congruent_mask = df_dual['flanker_condition'] == 'congruent'
        ssrt_congruent = compute_SSRT(df_dual, condition_mask=congruent_mask)
        
        # Test SSRT for incongruent condition only
        incongruent_mask = df_dual['flanker_condition'] == 'incongruent'
        ssrt_incongruent = compute_SSRT(df_dual, condition_mask=incongruent_mask)
        
        # Both should be valid SSRT values
        assert not np.isnan(ssrt_congruent)
        assert not np.isnan(ssrt_incongruent)
        
        # They might be different due to different trial counts
        assert isinstance(ssrt_congruent, (int, float))
        assert isinstance(ssrt_incongruent, (int, float))
        
    def test_compute_stop_signal_metrics_single(self):
        """Test complete stop signal metrics for single task."""
        metrics = compute_stop_signal_metrics(self.df, dual_task=False)
        
        # Check that all expected metrics are present
        expected_keys = ['go_rt', 'stop_fail_rt', 'go_acc', 'stop_fail_acc', 
                        'stop_success', 'avg_ssd', 'min_ssd', 'max_ssd', 
                        'min_ssd_count', 'max_ssd_count', 'ssrt']
        for key in expected_keys:
            assert key in metrics
        
    def test_compute_stop_signal_metrics_dual(self):
        """Test complete stop signal metrics for dual task."""
        # Create dual task data
        df_dual = pd.DataFrame({
            'SS_trial_type': ['go', 'go', 'stop', 'go', 'stop'],
            'correct_trial': [1, 1, 0, 1, 0],
            'rt': [0.5, 0.6, 0.7, 0.8, 0.9],
            'key_press': [1, 1, 2, 1, 2],
            'flanker_condition': ['congruent', 'congruent', 'congruent', 'incongruent', 'incongruent'],
            'SS_delay': [np.nan, np.nan, 0.2, np.nan, 0.3],
            'stim': ['A', 'B', 'A', 'B', 'A'],
            'correct_response': [1, 2, 1, 2, 1]
        })
        
        paired_conditions = ['congruent', 'incongruent']
        metrics = compute_stop_signal_metrics(
            df_dual, dual_task=True, paired_task_col='flanker_condition', 
            paired_conditions=paired_conditions, stim_cols=['stim']
        )
        
        # Check that metrics are calculated for each condition
        for condition in paired_conditions:
            assert f'{condition}_go_rt' in metrics
            assert f'{condition}_stop_fail_rt' in metrics
            assert f'{condition}_go_acc' in metrics
            assert f'{condition}_stop_fail_acc' in metrics
            assert f'{condition}_stop_success' in metrics
            assert f'{condition}_ssrt' in metrics  # Now includes SSRT for each condition
        
        # Check that global metrics are present (but no global SSRT for dual tasks)
        assert 'avg_ssd' in metrics
        # Note: Global SSRT is no longer calculated for dual tasks

if __name__ == "__main__":
    pytest.main([__file__]) 
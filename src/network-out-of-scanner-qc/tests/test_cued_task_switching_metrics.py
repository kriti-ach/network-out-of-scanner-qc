import pandas as pd
import numpy as np
import pytest
from utils.qc_utils import (
    compute_cued_task_switching_metrics,
    compute_cued_spatial_task_switching_metrics
)

class TestCuedTaskSwitchingMetrics:
    """Test cued task switching metric calculation functions."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample cued task switching data
        self.df = pd.DataFrame({
            'task_condition': ['stay', 'stay', 'switch', 'switch', 'stay', 'switch'],
            'cue_condition': ['stay', 'switch', 'stay', 'switch', 'stay', 'switch'],
            'correct_trial': [1, 1, 0, 1, 0, 1],
            'rt': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'key_press': [1, 1, 2, 1, 2, 1],
            'trial_id': ['test_trial'] * 6
        })
        
        # Create cued task switching with flanker data
        self.df_flanker = pd.DataFrame({
            'flanker_condition': ['congruent', 'incongruent', 'congruent', 'incongruent'],
            'task_condition': ['stay', 'stay', 'switch', 'switch'],
            'cue_condition': ['stay', 'switch', 'stay', 'switch'],
            'correct_trial': [1, 1, 0, 1],
            'rt': [0.5, 0.6, 0.7, 0.8],
            'key_press': [1, 1, 2, 1],
            'trial_id': ['test_trial'] * 4
        })
        
        # Create cued task switching with go/nogo data
        self.df_go_nogo = pd.DataFrame({
            'go_nogo_condition': ['go', 'nogo', 'go', 'nogo'],
            'task_condition': ['stay', 'stay', 'switch', 'switch'],
            'cue_condition': ['stay', 'switch', 'stay', 'switch'],
            'correct_trial': [1, 1, 0, 1],
            'rt': [0.5, 0.6, 0.7, 0.8],
            'key_press': [1, 1, 2, 1],
            'trial_id': ['test_trial'] * 4
        })
        
        # Create cued task switching with spatial task switching data
        self.df_spatial = pd.DataFrame({
            'task_condition': ['stay', 'stay', 'switch', 'switch'],
            'cue_condition': ['stay', 'switch', 'stay', 'switch'],
            'task_switch': ['tstay_cstay', 'tstay_cswitch', 'tswitch_cstay', 'tswitch_cswitch'],
            'correct_trial': [1, 1, 0, 1],
            'rt': [0.5, 0.6, 0.7, 0.8],
            'key_press': [1, 1, 2, 1],
            'trial_id': ['test_trial'] * 4
        })
        
        # Define condition lists
        self.single_conditions = ['tstay_cstay', 'tstay_cswitch', 'tswitch_cstay', 'tswitch_cswitch']
        self.flanker_conditions = ['congruent_tstay_cstay', 'incongruent_tstay_cswitch', 
                                  'congruent_tswitch_cstay', 'incongruent_tswitch_cswitch']
        self.go_nogo_conditions = ['go_tstay_cstay', 'nogo_tstay_cswitch', 
                                  'go_tswitch_cstay', 'nogo_tswitch_cswitch']
        self.spatial_conditions = ['cuedtstaycstay_spatialtstaycstay', 
                                  'cuedtstaycswitch_spatialtstaycswitch',
                                  'cuedtswitchcstay_spatialtswitchcstay',
                                  'cuedtswitchcswitch_spatialtswitchcswitch']
        
    def test_compute_cued_task_switching_metrics_single(self):
        """Test single cued task switching metrics calculation."""
        metrics = compute_cued_task_switching_metrics(
            self.df, self.single_conditions, 'single'
        )
        
        # Should calculate metrics for each condition
        for condition in self.single_conditions:
            assert f'{condition}_acc' in metrics
            assert f'{condition}_rt' in metrics
            assert f'{condition}_omission_rate' in metrics
            assert f'{condition}_commission_rate' in metrics
            
        # Check specific values for tstay_cstay (1 correct out of 2 trials)
        assert metrics['tstay_cstay_acc'] == pytest.approx(0.5)
        
        # Check RT for correct trials in tstay_cstay (mean of 0.5)
        assert metrics['tstay_cstay_rt'] == pytest.approx(0.5)
        
        # Check omission rate for tstay_cstay (0 omissions)
        assert metrics['tstay_cstay_omission_rate'] == 0.0
        
        # Check commission rate for tstay_cstay (1 commission out of 2 trials)
        assert metrics['tstay_cstay_commission_rate'] == pytest.approx(0.5)
        
    def test_compute_cued_task_switching_metrics_flanker(self):
        """Test cued task switching with flanker metrics calculation."""
        metrics = compute_cued_task_switching_metrics(
            self.df_flanker, self.flanker_conditions, 'flanker', flanker_col='flanker_condition'
        )
        
        # Should calculate metrics for each condition
        for condition in self.flanker_conditions:
            assert f'{condition}_acc' in metrics
            assert f'{condition}_rt' in metrics
            assert f'{condition}_omission_rate' in metrics
            assert f'{condition}_commission_rate' in metrics
            
        # Check specific values for congruent_tstay_cstay (1 correct out of 1 trial)
        assert metrics['congruent_tstay_cstay_acc'] == 1.0
        assert metrics['congruent_tstay_cstay_rt'] == pytest.approx(0.5)
        
    def test_compute_cued_task_switching_metrics_go_nogo(self):
        """Test cued task switching with go/nogo metrics calculation."""
        metrics = compute_cued_task_switching_metrics(
            self.df_go_nogo, self.go_nogo_conditions, 'go_nogo', go_nogo_col='go_nogo_condition'
        )
        
        # Should calculate metrics for each condition
        for condition in self.go_nogo_conditions:
            assert f'{condition}_acc' in metrics
            assert f'{condition}_rt' in metrics
            # For nogo conditions, omission/commission rates are not included
            if 'nogo' not in condition:
                assert f'{condition}_omission_rate' in metrics
                assert f'{condition}_commission_rate' in metrics
            
        # Check specific values for go_tstay_cstay (1 correct out of 1 trial)
        assert metrics['go_tstay_cstay_acc'] == 1.0
        assert metrics['go_tstay_cstay_rt'] == pytest.approx(0.5)
        
    def test_compute_cued_task_switching_metrics_case_sensitivity(self):
        """Test case sensitivity in condition matching."""
        # Test with mixed case in conditions
        df_mixed_case = pd.DataFrame({
            'task_condition': ['Stay', 'SWITCH', 'stay', 'Switch'],
            'cue_condition': ['STAY', 'switch', 'Stay', 'SWITCH'],
            'correct_trial': [1, 1, 1, 1],
            'rt': [0.5, 0.6, 0.7, 0.8],
            'key_press': [1, 1, 1, 1],
            'trial_id': ['test_trial'] * 4
        })
        
        metrics = compute_cued_task_switching_metrics(
            df_mixed_case, self.single_conditions, 'single'
        )
        
        # Should still calculate metrics (case-insensitive matching)
        for condition in self.single_conditions:
            assert f'{condition}_acc' in metrics
            
    def test_compute_cued_task_switching_metrics_with_nan_values(self):
        """Test cued task switching metrics with NaN values."""
        df_with_nan = pd.DataFrame({
            'task_condition': ['stay', np.nan, 'switch', 'stay'],
            'cue_condition': ['stay', 'switch', np.nan, 'switch'],
            'correct_trial': [1, 1, 1, 1],
            'rt': [0.5, 0.6, 0.7, 0.8],
            'key_press': [1, 1, 1, 1],
            'trial_id': ['test_trial'] * 4
        })
        
        metrics = compute_cued_task_switching_metrics(
            df_with_nan, self.single_conditions, 'single'
        )
        
        # Should still calculate metrics for valid conditions
        assert 'tstay_cstay_acc' in metrics
        assert 'tstay_cswitch_acc' in metrics
        
    def test_compute_cued_spatial_task_switching_metrics(self):
        """Test cued task switching with spatial task switching metrics."""
        metrics = compute_cued_spatial_task_switching_metrics(
            self.df_spatial, self.spatial_conditions
        )
        
        # Should calculate metrics for each condition
        for condition in self.spatial_conditions:
            assert f'{condition}_acc' in metrics
            assert f'{condition}_rt' in metrics
            assert f'{condition}_omission_rate' in metrics
            assert f'{condition}_commission_rate' in metrics
            
        # Check specific values for cuedtstaycstay_spatialtstaycstay (1 correct out of 1 trial)
        assert metrics['cuedtstaycstay_spatialtstaycstay_acc'] == 1.0
        assert metrics['cuedtstaycstay_spatialtstaycstay_rt'] == pytest.approx(0.5)
        
    def test_compute_cued_spatial_task_switching_metrics_complex_parsing(self):
        """Test complex condition parsing for cued + spatial task switching."""
        # Test with more complex condition names
        complex_conditions = [
            'cuedtstaycstay_spatialtstaycstay',
            'cuedtswitchcswitch_spatialtswitchcswitch'
        ]
        
        metrics = compute_cued_spatial_task_switching_metrics(
            self.df_spatial, complex_conditions
        )
        
        # Should parse and calculate metrics correctly
        for condition in complex_conditions:
            assert f'{condition}_acc' in metrics
            assert f'{condition}_rt' in metrics
            
    def test_compute_cued_spatial_task_switching_metrics_parsing_errors(self):
        """Test error handling for malformed condition names."""
        # Test with malformed condition names
        malformed_conditions = [
            'invalid_condition',
            'cuedtstaycstay_invalid',
            'cued_invalid_spatialtstaycstay'
        ]
        
        metrics = compute_cued_spatial_task_switching_metrics(
            self.df_spatial, malformed_conditions
        )
        
        # Should handle errors gracefully and skip malformed conditions
        assert len(metrics) == 0
        
    def test_condition_filtering(self):
        """Test filtering of invalid conditions."""
        # Test with 'na' values in conditions
        df_with_na = pd.DataFrame({
            'task_condition': ['stay', 'na', 'switch', 'stay'],
            'cue_condition': ['stay', 'switch', 'na', 'switch'],
            'correct_trial': [1, 1, 1, 1],
            'rt': [0.5, 0.6, 0.7, 0.8],
            'key_press': [1, 1, 1, 1],
            'trial_id': ['test_trial'] * 4
        })
        
        metrics = compute_cued_task_switching_metrics(
            df_with_na, self.single_conditions, 'single'
        )
        
        # Should only calculate metrics for valid conditions
        assert 'tstay_cstay_acc' in metrics
        assert 'tstay_cswitch_acc' in metrics
        
    def test_omission_and_commission_calculation(self):
        """Test omission and commission rate calculations."""
        # Create data with specific omission and commission patterns
        df_test = pd.DataFrame({
            'task_condition': ['stay', 'stay', 'switch', 'switch'],
            'cue_condition': ['stay', 'switch', 'stay', 'switch'],
            'correct_trial': [1, 0, 0, 1],
            'rt': [0.5, 0.6, 0.7, 0.8],
            'key_press': [1, -1, 2, 1],  # -1 is omission, 2 is commission
            'trial_id': ['test_trial'] * 4
        })
        
        metrics = compute_cued_task_switching_metrics(
            df_test, ['tstay_cstay', 'tstay_cswitch'], 'single'
        )
        
        # Check omission rate for tstay_cstay (0 omissions)
        assert metrics['tstay_cstay_omission_rate'] == 0.0
        
        # Check commission rate for tstay_cstay (0 commissions)
        assert metrics['tstay_cstay_commission_rate'] == 0.0
        
        # Check omission rate for tstay_cswitch (1 omission out of 1 trial)
        assert metrics['tstay_cswitch_omission_rate'] == 1.0
        
        # Check commission rate for tstay_cswitch (0 commissions)
        assert metrics['tstay_cswitch_commission_rate'] == 0.0
        
    def test_rt_calculation_correct_trials_only(self):
        """Test that RT is only calculated for correct trials."""
        df_test = pd.DataFrame({
            'task_condition': ['stay', 'stay', 'switch', 'switch'],
            'cue_condition': ['stay', 'switch', 'stay', 'switch'],
            'correct_trial': [1, 0, 1, 0],  # Only 1st and 3rd are correct
            'rt': [0.5, 0.6, 0.7, 0.8],
            'key_press': [1, 2, 1, 2],
            'trial_id': ['test_trial'] * 4
        })
        
        metrics = compute_cued_task_switching_metrics(
            df_test, ['tstay_cstay', 'tswitch_cstay'], 'single'
        )
        
        # RT should only include correct trials
        assert metrics['tstay_cstay_rt'] == pytest.approx(0.5)  # Only correct trial
        assert metrics['tswitch_cstay_rt'] == pytest.approx(0.7)  # Only correct trial

if __name__ == "__main__":
    pytest.main([__file__]) 
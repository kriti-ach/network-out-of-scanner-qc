import pandas as pd
import numpy as np
import pytest
from utils.exclusion_utils import (
    compare_to_threshold,
    append_exclusion_row,
    check_stop_signal_exclusion_criteria,
    check_go_nogo_exclusion_criteria,
    check_n_back_exclusion_criteria,
    check_other_exclusion_criteria,
    check_exclusion_criteria,
    suffix,
    prefix,
)
from utils.globals import (
    STOP_SUCCESS_ACC_LOW_THRESHOLD,
    STOP_SUCCESS_ACC_HIGH_THRESHOLD,
    GO_RT_THRESHOLD,
    GO_ACC_THRESHOLD_GO_NOGO,
    NOGO_ACC_THRESHOLD_GO_NOGO,
    ACC_THRESHOLD,
    OMISSION_RATE_THRESHOLD,
)


def test_compare_to_threshold():
    assert compare_to_threshold('go_acc', 0.5, ACC_THRESHOLD) == (0.5 < ACC_THRESHOLD)
    assert compare_to_threshold('stop_success_low', 0.2, STOP_SUCCESS_ACC_LOW_THRESHOLD) == (0.2 < STOP_SUCCESS_ACC_LOW_THRESHOLD)
    assert compare_to_threshold('stop_success_high', 0.8, STOP_SUCCESS_ACC_HIGH_THRESHOLD) == (0.8 > STOP_SUCCESS_ACC_HIGH_THRESHOLD)
    assert compare_to_threshold('go_rt', 2000, GO_RT_THRESHOLD) == (2000 > GO_RT_THRESHOLD)


def test_append_exclusion_row():
    df = pd.DataFrame({'subject_id': [], 'metric': [], 'metric_value': [], 'threshold': []})
    out = append_exclusion_row(df, 's01', 'go_acc', 0.5, ACC_THRESHOLD)
    assert len(out) == 1
    assert out.iloc[0]['subject_id'] == 's01'
    assert out.iloc[0]['metric'] == 'go_acc'


def test_check_stop_signal_exclusion_criteria_flags():
    # Two subjects, one with violations
    task_csv = pd.DataFrame({
        'subject_id': ['s01', 's02', 'mean', 'std', 'max', 'min'],
        'cued_go_rt': [GO_RT_THRESHOLD + 1, GO_RT_THRESHOLD - 1, np.nan, np.nan, np.nan, np.nan],
        'cued_go_acc': [ACC_THRESHOLD - 0.1, ACC_THRESHOLD + 0.1, np.nan, np.nan, np.nan, np.nan],
        'cued_stop_success': [STOP_SUCCESS_ACC_LOW_THRESHOLD - 0.01, STOP_SUCCESS_ACC_HIGH_THRESHOLD - 0.01, np.nan, np.nan, np.nan, np.nan],
        'cued_go_omission_rate': [OMISSION_RATE_THRESHOLD + 0.1, OMISSION_RATE_THRESHOLD - 0.1, np.nan, np.nan, np.nan, np.nan],
    })
    exclusion_df = pd.DataFrame({'subject_id': [], 'metric': [], 'metric_value': [], 'threshold': []})
    out = check_stop_signal_exclusion_criteria('stop_signal_with_cued_task_switching', task_csv, exclusion_df)
    # Expect flags for s01; none for s02; summary rows ignored
    assert (out['subject_id'] == 's01').any()
    assert not (out['subject_id'] == 's02').any()


def test_check_go_nogo_exclusion_criteria_flags():
    # Matching prefixes (e.g., "tstay_cstay_") so they pair
    task_csv = pd.DataFrame({
        'subject_id': ['s01', 's02', 'mean', 'std', 'max', 'min'],
        'tstay_cstay_go_acc': [GO_ACC_THRESHOLD_GO_NOGO - 0.05, GO_ACC_THRESHOLD_GO_NOGO + 0.05, np.nan, np.nan, np.nan, np.nan],
        'tstay_cstay_nogo_acc': [NOGO_ACC_THRESHOLD_GO_NOGO - 0.05, NOGO_ACC_THRESHOLD_GO_NOGO + 0.05, np.nan, np.nan, np.nan, np.nan],
        'tstay_cstay_go_omission_rate': [OMISSION_RATE_THRESHOLD + 0.2, OMISSION_RATE_THRESHOLD - 0.1, np.nan, np.nan, np.nan, np.nan],
    })
    exclusion_df = pd.DataFrame({'subject_id': [], 'metric': [], 'metric_value': [], 'threshold': []})
    out = check_go_nogo_exclusion_criteria('flanker_with_go_nogo', task_csv, exclusion_df)
    # s01 should have multiple flags; s02 should have none
    assert (out['subject_id'] == 's01').any()
    assert not (out['subject_id'] == 's02').any()


def test_check_n_back_exclusion_criteria_independent_flags():
    # Construct minimal columns for level 1
    task_csv = pd.DataFrame({
        'subject_id': ['s01', 'mean', 'std', 'max', 'min'],
        'mismatch_1.0back_acc_cstay': [ACC_THRESHOLD - 0.1, np.nan, np.nan, np.nan, np.nan],
        'match_1.0back_acc_cstay': [ACC_THRESHOLD - 0.05, np.nan, np.nan, np.nan, np.nan],
        'mismatch_1.0back_omission_rate_cstay': [OMISSION_RATE_THRESHOLD + 0.2, np.nan, np.nan, np.nan, np.nan],
        'match_1.0back_omission_rate_cstay': [OMISSION_RATE_THRESHOLD + 0.1, np.nan, np.nan, np.nan, np.nan],
    })
    exclusion_df = pd.DataFrame({'subject_id': [], 'metric': [], 'metric_value': [], 'threshold': []})
    out = check_n_back_exclusion_criteria('n_back_with_cued_task_switching', task_csv, exclusion_df)
    assert (out['subject_id'] == 's01').any()


def test_check_other_exclusion_criteria_generic():
    task_csv = pd.DataFrame({
        'subject_id': ['s01', 'mean', 'std', 'max', 'min'],
        'some_acc': [ACC_THRESHOLD - 0.01, np.nan, np.nan, np.nan, np.nan],
        'some_omission_rate': [OMISSION_RATE_THRESHOLD + 0.01, np.nan, np.nan, np.nan, np.nan],
        'some_rt': [100, np.nan, np.nan, np.nan, np.nan],
    })
    exclusion_df = pd.DataFrame({'subject_id': [], 'metric': [], 'metric_value': [], 'threshold': []})
    out = check_other_exclusion_criteria('flanker', task_csv, exclusion_df)
    assert (out['metric'] == 'some_acc').any()
    assert (out['metric'] == 'some_omission_rate').any()


def test_check_exclusion_criteria_router():
    # Create a CSV with columns that will trigger each branch minimally
    task_csv = pd.DataFrame({
        'subject_id': ['s01', 'mean', 'std', 'max', 'min'],
        'go_rt': [GO_RT_THRESHOLD + 10, np.nan, np.nan, np.nan, np.nan],
        'go_acc': [ACC_THRESHOLD - 0.1, np.nan, np.nan, np.nan, np.nan],
        'stop_success': [STOP_SUCCESS_ACC_LOW_THRESHOLD - 0.01, np.nan, np.nan, np.nan, np.nan],
        'go_omission_rate': [OMISSION_RATE_THRESHOLD + 0.2, np.nan, np.nan, np.nan, np.nan],
        'tstay_cstay_go_acc': [GO_ACC_THRESHOLD_GO_NOGO - 0.1, np.nan, np.nan, np.nan, np.nan],
        'tstay_cstay_nogo_acc': [NOGO_ACC_THRESHOLD_GO_NOGO - 0.1, np.nan, np.nan, np.nan, np.nan],
        'match_1.0back_acc_cstay': [ACC_THRESHOLD - 0.1, np.nan, np.nan, np.nan, np.nan],
        'mismatch_1.0back_acc_cstay': [ACC_THRESHOLD - 0.1, np.nan, np.nan, np.nan, np.nan],
        'some_acc': [ACC_THRESHOLD - 0.01, np.nan, np.nan, np.nan, np.nan],
    })
    exclusion_df = pd.DataFrame({'subject_id': [], 'metric': [], 'metric_value': [], 'threshold': []})
    out = check_exclusion_criteria('stop_signal_with_go_nogo_and_n_back_and_flanker', task_csv, exclusion_df)
    # Should have at least one row from several branches
    assert len(out[out['subject_id'] == 's01']) >= 1


def test_suffix_basic_extraction():
    # Extract everything after the prefix, preserving condition details
    col = 'match_1.0back_tstay_cstay_acc'
    pref = 'match_1.0back_'
    assert suffix(col, pref) == 'tstay_cstay_acc'


def test_suffix_prefix_not_found_returns_original():
    col = 'mismatch_2.0back_incongruent_acc'
    pref = 'not_present_'
    assert suffix(col, pref) == col


def test_suffix_complex_suffix_parts():
    col = 'mismatch_3.0back_congruent_tstay_cswitch_omission_rate'
    pref = 'mismatch_3.0back_'
    assert suffix(col, pref) == 'congruent_tstay_cswitch_omission_rate'


def test_nback_dual_ignores_non_nback_accuracy_and_checks_others():
    # Task includes n_back; accuracy should be driven by n-back only, but other metrics (e.g., omission) still apply                                                                                              
    task_csv = pd.DataFrame({
        'subject_id': ['s01', 'mean', 'std', 'max', 'min'],
        'go_rt': [GO_RT_THRESHOLD + 10, np.nan, np.nan, np.nan, np.nan],
        'go_acc': [ACC_THRESHOLD - 0.2, np.nan, np.nan, np.nan, np.nan],  # Should be ignored due to n_back                                                                                                       
        'go_omission_rate': [OMISSION_RATE_THRESHOLD + 0.2, np.nan, np.nan, np.nan, np.nan],  # Should be checked                                                                                                 
        'mismatch_1.0back_acc_cstay': [ACC_THRESHOLD - 0.2, np.nan, np.nan, np.nan, np.nan],
        'match_1.0back_acc_cstay': [ACC_THRESHOLD - 0.1, np.nan, np.nan, np.nan, np.nan],
    })
    exclusion_df = pd.DataFrame({'subject_id': [], 'metric': [], 'metric_value': [], 'threshold': []})
    out = check_exclusion_criteria('n_back_with_flanker', task_csv, exclusion_df)
    # Should include nback accuracy flags and omission_rate, but not generic go_acc
    assert (out['metric'].str.contains('mismatch_1.0back_acc').any())
    assert (out['metric'].str.contains('match_1.0back_acc').any())
    assert (out['metric'] == 'go_omission_rate').any()
    assert not (out['metric'] == 'go_acc').any()


def test_gng_nback_prioritization_nogo_and_nback_go_only():
    # In n_back + go_nogo: prioritize nogo_acc for nogo, and nback thresholds for go condition
    task_csv = pd.DataFrame({
        'subject_id': ['s01', 'mean', 'std', 'max', 'min'],
        'tstay_cstay_go_acc': [GO_ACC_THRESHOLD_GO_NOGO - 0.2, np.nan, np.nan, np.nan, np.nan],  # Should be ignored
        'tstay_cstay_nogo_acc': [NOGO_ACC_THRESHOLD_GO_NOGO - 0.1, np.nan, np.nan, np.nan, np.nan],  # Should be flagged
        # N-back columns (note: ensure no 'nogo' in these column names to be considered by nback checks)
        'mismatch_1.0back_acc_cstay': [ACC_THRESHOLD - 0.2, np.nan, np.nan, np.nan, np.nan],
        'match_1.0back_acc_cstay': [ACC_THRESHOLD - 0.1, np.nan, np.nan, np.nan, np.nan],
        # A column that looks like nback but includes 'nogo' should be ignored by nback logic
        'mismatch_1.0back_acc_cstay_nogo': [ACC_THRESHOLD - 0.5, np.nan, np.nan, np.nan, np.nan],
    })
    exclusion_df = pd.DataFrame({'subject_id': [], 'metric': [], 'metric_value': [], 'threshold': []})
    out = check_exclusion_criteria('go_nogo_with_n_back', task_csv, exclusion_df)
    # nogo_acc flagged
    assert (out['metric'] == 'tstay_cstay_nogo_acc').any()
    # go_acc not flagged because n_back owns go accuracy
    assert not (out['metric'] == 'tstay_cstay_go_acc').any()
    # nback match/mismatch flagged
    assert (out['metric'].str.contains('mismatch_1.0back_acc').any())
    assert (out['metric'].str.contains('match_1.0back_acc').any())
    # The fake nback column with 'nogo' in name should not produce a separate flag from nback logic
    assert not (out['metric'] == 'mismatch_1.0back_acc_cstay_nogo').any()


def test_other_exclusion_skips_accuracy_when_nback_task():
    # When n_back in task, generic accuracy in other_exclusion should be skipped, but omission should still be processed
    task_csv = pd.DataFrame({
        'subject_id': ['s01', 'mean', 'std', 'max', 'min'],
        'some_acc': [ACC_THRESHOLD - 0.2, np.nan, np.nan, np.nan, np.nan],
        'mismatch_1.0back_acc_cstay': [ACC_THRESHOLD - 0.2, np.nan, np.nan, np.nan, np.nan],
        'match_1.0back_acc_cstay': [ACC_THRESHOLD - 0.1, np.nan, np.nan, np.nan, np.nan],
    })
    exclusion_df = pd.DataFrame({'subject_id': [], 'metric': [], 'metric_value': [], 'threshold': []})
    out = check_exclusion_criteria('n_back_with_flanker', task_csv, exclusion_df)
    # some_acc should be ignored in other_exclusion for nback tasks
    assert not (out['metric'] == 'some_acc').any()
    # nback accuracies flagged
    assert (out['metric'].str.contains('mismatch_1.0back_acc').any())
    assert (out['metric'].str.contains('match_1.0back_acc').any())

def test_prefix_basic_extraction():
    col = 'stop_fail_rt_tstay_cstay'
    pref = 'tstay_cstay'
    assert prefix(col, pref) == 'stop_fail_rt_'


def test_stop_nback_collapsed_metrics():
    """Test that stop+nback tasks collapse ALL metrics across load levels."""
    # Create test data with multiple load levels for stop signal and N-back metrics
    task_csv = pd.DataFrame({
        'subject_id': ['s01', 'mean', 'std', 'max', 'min'],
        # Stop signal metrics across load levels
        '1.0back_stop_success': [0.2, np.nan, np.nan, np.nan, np.nan],  # Below threshold
        '2.0back_stop_success': [0.3, np.nan, np.nan, np.nan, np.nan],  # Below threshold
        '3.0back_stop_success': [0.1, np.nan, np.nan, np.nan, np.nan],  # Below threshold
        '1.0back_go_rt': [900, np.nan, np.nan, np.nan, np.nan],  # Above threshold
        '2.0back_go_rt': [950, np.nan, np.nan, np.nan, np.nan],  # Above threshold
        '3.0back_go_rt': [1000, np.nan, np.nan, np.nan, np.nan],  # Above threshold
        '1.0back_go_omission_rate': [0.3, np.nan, np.nan, np.nan, np.nan],  # Above threshold
        '2.0back_go_omission_rate': [0.4, np.nan, np.nan, np.nan, np.nan],  # Above threshold
        '3.0back_go_omission_rate': [0.5, np.nan, np.nan, np.nan, np.nan],  # Above threshold
        '1.0back_stop_fail_rt': [1200, np.nan, np.nan, np.nan, np.nan],  # Above go_rt
        '2.0back_stop_fail_rt': [1300, np.nan, np.nan, np.nan, np.nan],  # Above go_rt
        '3.0back_stop_fail_rt': [1400, np.nan, np.nan, np.nan, np.nan],  # Above go_rt
        # N-back specific metrics
        'mismatch_1.0back_congruent_acc': [0.6, np.nan, np.nan, np.nan, np.nan],  # Below 70%
        'mismatch_2.0back_congruent_acc': [0.5, np.nan, np.nan, np.nan, np.nan],  # Below 70%
        'mismatch_3.0back_congruent_acc': [0.4, np.nan, np.nan, np.nan, np.nan],  # Below 70%
        'match_1.0back_congruent_acc': [0.15, np.nan, np.nan, np.nan, np.nan],    # Below 20%
        'match_2.0back_congruent_acc': [0.1, np.nan, np.nan, np.nan, np.nan],     # Below 20%
        'match_3.0back_congruent_acc': [0.05, np.nan, np.nan, np.nan, np.nan],    # Below 20%
    })
    
    exclusion_df = pd.DataFrame({'subject_id': [], 'metric': [], 'metric_value': [], 'threshold': []})
    out = check_exclusion_criteria('stop_signal_n_back', task_csv, exclusion_df)
    
    # Should have collapsed stop signal metrics
    assert (out['metric'] == 'collapsed_stop_success_low').any()
    assert (out['metric'] == 'collapsed_go_rt').any()
    assert (out['metric'] == 'collapsed_go_omission_rate').any()
    assert (out['metric'] == 'collapsed_stop_fail_rt_greater_than_go_rt').any()
    
    # Should have collapsed N-back accuracy metrics
    assert (out['metric'] == 'collapsed_mismatch_acc_combined').any()
    assert (out['metric'] == 'collapsed_match_acc_combined').any()
    assert (out['metric'] == 'collapsed_mismatch_acc').any()
    assert (out['metric'] == 'collapsed_match_acc').any()
    
    # Check that collapsed values are means across loads
    stop_success_row = out[out['metric'] == 'collapsed_stop_success_low'].iloc[0]
    expected_stop_success = (0.2 + 0.3 + 0.1) / 3  # Mean across loads 1, 2, 3
    assert stop_success_row['metric_value'] == pytest.approx(expected_stop_success)
    
    go_rt_row = out[out['metric'] == 'collapsed_go_rt'].iloc[0]
    expected_go_rt = (900 + 950 + 1000) / 3  # Mean across loads 1, 2, 3
    assert go_rt_row['metric_value'] == pytest.approx(expected_go_rt)
    
    go_omission_row = out[out['metric'] == 'collapsed_go_omission_rate'].iloc[0]
    expected_go_omission = (0.3 + 0.4 + 0.5) / 3  # Mean across loads 1, 2, 3
    assert go_omission_row['metric_value'] == pytest.approx(expected_go_omission)
    
    # Check stop_fail_rt > go_rt comparison
    stop_fail_rt_row = out[out['metric'] == 'collapsed_stop_fail_rt_greater_than_go_rt'].iloc[0]
    expected_stop_fail_rt = (1200 + 1300 + 1400) / 3  # Mean across loads 1, 2, 3
    assert stop_fail_rt_row['metric_value'] == pytest.approx(expected_stop_fail_rt)
    assert stop_fail_rt_row['threshold'] == pytest.approx(expected_go_rt)  # threshold should be go_rt

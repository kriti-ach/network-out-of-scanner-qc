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
    remove_some_flags_for_exclusion,
    create_combined_exclusions_csv,
    flag_fmri_condition_metrics,
)
from utils.globals import (
    STOP_SUCCESS_ACC_LOW_THRESHOLD,
    STOP_SUCCESS_ACC_HIGH_THRESHOLD,
    GO_RT_THRESHOLD,
    GO_RT_THRESHOLD_FMRI,
    GO_ACC_THRESHOLD_GO_NOGO,
    NOGO_ACC_THRESHOLD_GO_NOGO,
    ACC_THRESHOLD,
    OMISSION_RATE_THRESHOLD,
    MATCH_THRESHOLD,
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


def test_append_exclusion_row_with_session():
    """Test append_exclusion_row with session parameter."""
    df = pd.DataFrame({'subject_id': [], 'metric': [], 'metric_value': [], 'threshold': []})
    out = append_exclusion_row(df, 's01', 'go_acc', 0.5, ACC_THRESHOLD, session='ses-01')
    assert len(out) == 1
    assert out.iloc[0]['subject_id'] == 's01'
    assert out.iloc[0]['session'] == 'ses-01'
    assert 'session' in out.columns
    assert out.columns.tolist() == ['subject_id', 'session', 'metric', 'metric_value', 'threshold']


def test_append_exclusion_row_prevents_duplicates():
    """Test that append_exclusion_row prevents duplicate entries."""
    df = pd.DataFrame({
        'subject_id': ['s01'],
        'metric': ['go_acc'],
        'metric_value': [0.5],
        'threshold': [ACC_THRESHOLD]
    })
    out = append_exclusion_row(df, 's01', 'go_acc', 0.6, ACC_THRESHOLD)
    # Should not add duplicate
    assert len(out) == 1
    assert out.iloc[0]['metric_value'] == 0.5  # Original value preserved


def test_append_exclusion_row_prevents_duplicates_with_session():
    """Test duplicate prevention with session."""
    df = pd.DataFrame({
        'subject_id': ['s01'],
        'session': ['ses-01'],
        'metric': ['go_acc'],
        'metric_value': [0.5],
        'threshold': [ACC_THRESHOLD]
    })
    out = append_exclusion_row(df, 's01', 'go_acc', 0.6, ACC_THRESHOLD, session='ses-01')
    assert len(out) == 1
    # Different session should allow new entry
    out2 = append_exclusion_row(df, 's01', 'go_acc', 0.6, ACC_THRESHOLD, session='ses-02')
    assert len(out2) == 2


def test_remove_some_flags_for_exclusion_stop_fail_rt():
    """Test removal of stop_fail_rt_greater_than_go_rt flags."""
    exclusion_df = pd.DataFrame({
        'subject_id': ['s01', 's02'],
        'metric': ['stop_fail_rt_greater_than_go_rt', 'go_acc'],
        'metric_value': [100, 0.4],
        'threshold': [50, ACC_THRESHOLD]
    })
    out = remove_some_flags_for_exclusion('stop_signal', exclusion_df)
    assert len(out) == 1
    assert out.iloc[0]['metric'] == 'go_acc'
    assert 'stop_fail_rt_greater_than_go_rt' not in out['metric'].values


def test_remove_some_flags_for_exclusion_3back():
    """Test removal of 3.0back flags."""
    exclusion_df = pd.DataFrame({
        'subject_id': ['s01', 's02'],
        'metric': ['match_3.0back_acc', 'match_1.0back_acc'],
        'metric_value': [0.1, 0.2],
        'threshold': [MATCH_THRESHOLD, MATCH_THRESHOLD]
    })
    out = remove_some_flags_for_exclusion('n_back', exclusion_df)
    assert len(out) == 1
    assert out.iloc[0]['metric'] == 'match_1.0back_acc'
    assert 'match_3.0back_acc' not in out['metric'].values


def test_remove_some_flags_for_exclusion_stop_nback():
    """Test removal of stop_success flags in stop+nback tasks."""
    exclusion_df = pd.DataFrame({
        'subject_id': ['s01', 's02'],
        'metric': ['stop_success', 'stop_success_collapsed'],
        'metric_value': [0.1, 0.2],
        'threshold': [STOP_SUCCESS_ACC_LOW_THRESHOLD, STOP_SUCCESS_ACC_LOW_THRESHOLD]
    })
    out = remove_some_flags_for_exclusion('stop_signal_with_n_back', exclusion_df)
    # Should keep stop_success_collapsed, remove stop_success
    assert len(out) == 1
    assert out.iloc[0]['metric'] == 'stop_success_collapsed'


def test_check_stop_signal_exclusion_criteria_fmri_mode():
    """Test stop signal exclusion criteria in fMRI mode."""
    task_csv = pd.DataFrame({
        'subject_id': ['s01', 's02', 'mean', 'std', 'max', 'min'],
        'session': ['ses-01', 'ses-01', '', '', '', ''],
        'overall_go_acc': [ACC_THRESHOLD - 0.1, ACC_THRESHOLD + 0.1, np.nan, np.nan, np.nan, np.nan],
        'overall_go_rt': [GO_RT_THRESHOLD_FMRI + 10, GO_RT_THRESHOLD_FMRI - 10, np.nan, np.nan, np.nan, np.nan],
        'overall_stop_success': [STOP_SUCCESS_ACC_LOW_THRESHOLD - 0.01, STOP_SUCCESS_ACC_HIGH_THRESHOLD + 0.01, np.nan, np.nan, np.nan, np.nan],
    })
    exclusion_df = pd.DataFrame({'subject_id': [], 'metric': [], 'metric_value': [], 'threshold': []})
    out = check_stop_signal_exclusion_criteria('stop_signal_with_flanker', task_csv, exclusion_df)
    # s01 should have flags; s02 should not
    assert (out['subject_id'] == 's01').any()
    assert 'session' in out.columns


def test_check_go_nogo_exclusion_criteria_fmri_mode():
    """Test go/nogo exclusion criteria in fMRI mode with new rules."""
    from utils.globals import GONOGO_GO_ACC_THRESHOLD_1, GONOGO_NOGO_ACC_THRESHOLD_1, GONOGO_GO_ACC_THRESHOLD_2, GONOGO_NOGO_ACC_THRESHOLD_2
    task_csv = pd.DataFrame({
        'subject_id': ['s01', 's02', 'mean', 'std', 'max', 'min'],
        'session': ['ses-01', 'ses-01', '', '', '', ''],
        'tstay_cstay_go_acc': [GONOGO_GO_ACC_THRESHOLD_1 - 0.1, GONOGO_GO_ACC_THRESHOLD_1 + 0.1, np.nan, np.nan, np.nan, np.nan],
        'tstay_cstay_nogo_acc': [GONOGO_NOGO_ACC_THRESHOLD_1 - 0.1, GONOGO_NOGO_ACC_THRESHOLD_1 + 0.1, np.nan, np.nan, np.nan, np.nan],
    })
    exclusion_df = pd.DataFrame({'subject_id': [], 'metric': [], 'metric_value': [], 'threshold': []})
    out = check_go_nogo_exclusion_criteria('go_nogo_with_flanker', task_csv, exclusion_df)
    # s01 should trigger both rules
    assert (out['subject_id'] == 's01').any()
    assert any('fmri_rule1' in str(m) for m in out['metric'].values) or any('fmri_rule2' in str(m) for m in out['metric'].values)


def test_check_n_back_exclusion_criteria_fmri_single():
    """Test n-back exclusion criteria for fMRI single task."""
    from utils.globals import NBACK_1BACK_MATCH_ACC_COMBINED_THRESHOLD_1, NBACK_1BACK_MISMATCH_ACC_COMBINED_THRESHOLD_1
    task_csv = pd.DataFrame({
        'subject_id': ['s01', 'mean', 'std', 'max', 'min'],
        'session': ['ses-01', '', '', '', ''],
        'match_1.0back_acc_cstay': [NBACK_1BACK_MATCH_ACC_COMBINED_THRESHOLD_1 - 0.1, np.nan, np.nan, np.nan, np.nan],
        'mismatch_1.0back_acc_cstay': [NBACK_1BACK_MISMATCH_ACC_COMBINED_THRESHOLD_1 - 0.1, np.nan, np.nan, np.nan, np.nan],
    })
    exclusion_df = pd.DataFrame({'subject_id': [], 'metric': [], 'metric_value': [], 'threshold': []})
    out = check_n_back_exclusion_criteria('n_back_single_task_network', task_csv, exclusion_df)
    assert (out['subject_id'] == 's01').any()
    assert any('fmri_rule1' in str(m) for m in out['metric'].values) or any('fmri_rule2' in str(m) for m in out['metric'].values)


def test_check_n_back_exclusion_criteria_fmri_dual():
    """Test n-back exclusion criteria for fMRI dual task."""
    from utils.globals import NBACK_1BACK_MATCH_ACC_COMBINED_THRESHOLD_1, NBACK_1BACK_MISMATCH_ACC_COMBINED_THRESHOLD_1
    task_csv = pd.DataFrame({
        'subject_id': ['s01', 'mean', 'std', 'max', 'min'],
        'session': ['ses-01', '', '', '', ''],
        'overall_match_1.0back_acc': [NBACK_1BACK_MATCH_ACC_COMBINED_THRESHOLD_1 - 0.1, np.nan, np.nan, np.nan, np.nan],
        'overall_mismatch_1.0back_acc': [NBACK_1BACK_MISMATCH_ACC_COMBINED_THRESHOLD_1 - 0.1, np.nan, np.nan, np.nan, np.nan],
    })
    exclusion_df = pd.DataFrame({'subject_id': [], 'metric': [], 'metric_value': [], 'threshold': []})
    out = check_n_back_exclusion_criteria('n_back_with_flanker', task_csv, exclusion_df)
    assert (out['subject_id'] == 's01').any()


def test_check_other_exclusion_criteria_fmri_mode():
    """Test other exclusion criteria in fMRI mode (overall_acc only)."""
    task_csv = pd.DataFrame({
        'subject_id': ['s01', 's02', 'mean', 'std', 'max', 'min'],
        'session': ['ses-01', 'ses-01', '', '', '', ''],
        'overall_acc': [ACC_THRESHOLD - 0.1, ACC_THRESHOLD + 0.1, np.nan, np.nan, np.nan, np.nan],
        'congruent_acc': [ACC_THRESHOLD - 0.1, ACC_THRESHOLD + 0.1, np.nan, np.nan, np.nan, np.nan],
    })
    exclusion_df = pd.DataFrame({'subject_id': [], 'metric': [], 'metric_value': [], 'threshold': []})
    out = check_other_exclusion_criteria('flanker', task_csv, exclusion_df)
    # Should only flag overall_acc, not congruent_acc
    assert (out['metric'] == 'overall_acc').any()
    assert not (out['metric'] == 'congruent_acc').any()


def test_flag_fmri_condition_metrics():
    """Test flag_fmri_condition_metrics function."""
    task_csv = pd.DataFrame({
        'subject_id': ['s01', 's02', 'mean', 'std', 'max', 'min'],
        'session': ['ses-01', 'ses-01', '', '', '', ''],
        'congruent_acc': [ACC_THRESHOLD - 0.1, ACC_THRESHOLD + 0.1, np.nan, np.nan, np.nan, np.nan],
        'incongruent_acc': [ACC_THRESHOLD - 0.1, ACC_THRESHOLD + 0.1, np.nan, np.nan, np.nan, np.nan],
        'congruent_omission_rate': [OMISSION_RATE_THRESHOLD + 0.1, OMISSION_RATE_THRESHOLD - 0.1, np.nan, np.nan, np.nan, np.nan],
    })
    condition_acc_flags, omission_rate_flags = flag_fmri_condition_metrics('flanker', task_csv)
    # s01 should have flags for both accuracies and omission rate
    assert len(condition_acc_flags) > 0
    assert len(omission_rate_flags) > 0
    assert 'session' in condition_acc_flags.columns
    assert 'session' in omission_rate_flags.columns


def test_flag_fmri_condition_metrics_nback():
    """Test flag_fmri_condition_metrics for n-back tasks."""
    from utils.globals import MATCH_THRESHOLD, MISMATCH_THRESHOLD
    task_csv = pd.DataFrame({
        'subject_id': ['s01', 'mean', 'std', 'max', 'min'],
        'session': ['ses-01', '', '', '', ''],
        'match_1.0back_acc_cstay': [MATCH_THRESHOLD - 0.1, np.nan, np.nan, np.nan, np.nan],
        'mismatch_1.0back_acc_cstay': [MISMATCH_THRESHOLD - 0.1, np.nan, np.nan, np.nan, np.nan],
    })
    condition_acc_flags, omission_rate_flags = flag_fmri_condition_metrics('n_back_with_flanker', task_csv)
    assert len(condition_acc_flags) > 0
    assert any('match_1.0back_acc' in str(m) for m in condition_acc_flags['metric'].values)
    assert any('mismatch_1.0back_acc' in str(m) for m in condition_acc_flags['metric'].values)


def test_flag_fmri_condition_metrics_go_nogo():
    """Test flag_fmri_condition_metrics for go/nogo tasks."""
    task_csv = pd.DataFrame({
        'subject_id': ['s01', 'mean', 'std', 'max', 'min'],
        'session': ['ses-01', '', '', '', ''],
        'tstay_cstay_go_acc': [GO_ACC_THRESHOLD_GO_NOGO - 0.1, np.nan, np.nan, np.nan, np.nan],
        'tstay_cstay_nogo_acc': [NOGO_ACC_THRESHOLD_GO_NOGO - 0.1, np.nan, np.nan, np.nan, np.nan],
    })
    condition_acc_flags, omission_rate_flags = flag_fmri_condition_metrics('go_nogo_with_flanker', task_csv)
    assert len(condition_acc_flags) > 0
    assert any('go_acc' in m for m in condition_acc_flags['metric'].values)
    assert any('nogo_acc' in m for m in condition_acc_flags['metric'].values)


def test_create_combined_exclusions_csv(tmp_path):
    """Test create_combined_exclusions_csv function."""
    exclusions_output_path = tmp_path / 'exclusions'
    exclusions_output_path.mkdir()
    
    # Create sample exclusion files
    task1_exclusions = pd.DataFrame({
        'subject_id': ['s01', 's02'],
        'metric': ['go_acc', 'go_rt'],
        'metric_value': [0.4, 1000],
        'threshold': [ACC_THRESHOLD, GO_RT_THRESHOLD]
    })
    task1_exclusions.to_csv(exclusions_output_path / 'excluded_data_flanker.csv', index=False)
    
    task2_exclusions = pd.DataFrame({
        'subject_id': ['s01', 's03'],
        'metric': ['stop_success', 'go_acc'],
        'metric_value': [0.1, 0.3],
        'threshold': [STOP_SUCCESS_ACC_LOW_THRESHOLD, ACC_THRESHOLD]
    })
    task2_exclusions.to_csv(exclusions_output_path / 'excluded_data_stop_signal.csv', index=False)
    
    tasks = ['flanker', 'stop_signal']
    create_combined_exclusions_csv(tasks, exclusions_output_path)
    
    # Check that combined files were created
    assert (exclusions_output_path / 'all_exclusions.csv').exists()
    assert (exclusions_output_path / 'summarized_exclusions.csv').exists()
    
    # Check contents
    all_exclusions = pd.read_csv(exclusions_output_path / 'all_exclusions.csv')
    assert len(all_exclusions) == 4  # 2 + 2
    assert 'task_name' in all_exclusions.columns
    assert set(all_exclusions['task_name'].unique()) == {'flanker', 'stop_signal'}
    
    summarized = pd.read_csv(exclusions_output_path / 'summarized_exclusions.csv')
    assert len(summarized) == 4  # s01-flanker, s01-stop_signal, s02-flanker, s03-stop_signal
    assert len(summarized[summarized['subject_id'] == 's01']) == 2  # s01 appears in both tasks
    assert set(summarized['subject_id'].unique()) == {'s01', 's02', 's03'}


def test_create_combined_exclusions_csv_with_session(tmp_path):
    """Test create_combined_exclusions_csv with session column."""
    exclusions_output_path = tmp_path / 'exclusions'
    exclusions_output_path.mkdir()
    
    task1_exclusions = pd.DataFrame({
        'subject_id': ['s01', 's01'],
        'session': ['ses-01', 'ses-02'],
        'metric': ['go_acc', 'go_acc'],
        'metric_value': [0.4, 0.3],
        'threshold': [ACC_THRESHOLD, ACC_THRESHOLD]
    })
    task1_exclusions.to_csv(exclusions_output_path / 'excluded_data_flanker.csv', index=False)
    
    tasks = ['flanker']
    create_combined_exclusions_csv(tasks, exclusions_output_path)
    
    all_exclusions = pd.read_csv(exclusions_output_path / 'all_exclusions.csv')
    assert 'session' in all_exclusions.columns
    assert len(all_exclusions) == 2
    
    summarized = pd.read_csv(exclusions_output_path / 'summarized_exclusions.csv')
    assert 'session' in summarized.columns
    assert len(summarized) == 2  # Two different sessions


def test_check_stop_signal_stop_fail_rt_greater_than_go_rt():
    """Test stop_fail_rt > go_rt exclusion check."""
    task_csv = pd.DataFrame({
        'subject_id': ['s01', 's02', 'mean', 'std', 'max', 'min'],
        'cued_go_rt': [500, 500, np.nan, np.nan, np.nan, np.nan],
        'cued_stop_fail_rt': [600, 400, np.nan, np.nan, np.nan, np.nan],  # s01 violates, s02 doesn't
    })
    exclusion_df = pd.DataFrame({'subject_id': [], 'metric': [], 'metric_value': [], 'threshold': []})
    out = check_stop_signal_exclusion_criteria('stop_signal_with_cued_task_switching', task_csv, exclusion_df)
    # s01 should have stop_fail_rt_greater_than_go_rt flag
    assert any('stop_fail_rt_greater_than_go_rt' in m for m in out[out['subject_id'] == 's01']['metric'].values)
    assert not any('stop_fail_rt_greater_than_go_rt' in m for m in out[out['subject_id'] == 's02']['metric'].values)


def test_check_go_nogo_single_task():
    """Test go/nogo exclusion for single task (different thresholds)."""
    task_csv = pd.DataFrame({
        'subject_id': ['s01', 's02', 'mean', 'std', 'max', 'min'],
        'go_acc': [GO_ACC_THRESHOLD_GO_NOGO - 0.1, GO_ACC_THRESHOLD_GO_NOGO + 0.1, np.nan, np.nan, np.nan, np.nan],
        'nogo_acc': [NOGO_ACC_THRESHOLD_GO_NOGO - 0.1, NOGO_ACC_THRESHOLD_GO_NOGO + 0.1, np.nan, np.nan, np.nan, np.nan],
    })
    exclusion_df = pd.DataFrame({'subject_id': [], 'metric': [], 'metric_value': [], 'threshold': []})
    out = check_go_nogo_exclusion_criteria('go_nogo_single_task_network', task_csv, exclusion_df)
    assert (out['subject_id'] == 's01').any()
    assert not (out['subject_id'] == 's02').any()


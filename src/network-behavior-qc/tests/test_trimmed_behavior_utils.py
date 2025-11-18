import pandas as pd
import numpy as np
import pytest
from utils.trimmed_behavior_utils import preprocess_rt_tail_cutoff, get_bids_task_name


def test_preprocess_rt_tail_cutoff_no_trimming_needed():
    """Test when no trimming is needed - all responses are valid."""
    df = pd.DataFrame({
        'trial_id': ['test_trial'] * 10 + ['fixation'] * 5,
        'rt': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] + [-1] * 5
    })
    df_trimmed, cut_pos, cut_before_halfway, proportion_blank = preprocess_rt_tail_cutoff(
        df, subject_id='s01', task_name='flanker'
    )
    assert df_trimmed.equals(df)
    assert cut_pos is None
    assert cut_before_halfway is False
    assert proportion_blank == 0.0


def test_preprocess_rt_tail_cutoff_trim_trailing_blanks():
    """Test trimming when there are trailing blank trials."""
    # 10 test trials: first 7 have RTs, last 3 are blank (-1)
    df = pd.DataFrame({
        'trial_id': ['test_trial'] * 10 + ['fixation'] * 3,
        'rt': [100, 200, 300, 400, 500, 600, 700, -1, -1, -1] + [-1] * 3
    })
    df_trimmed, cut_pos, cut_before_halfway, proportion_blank = preprocess_rt_tail_cutoff(
        df, subject_id='s01', task_name='flanker', last_n_test_trials=3
    )
    # Should trim to first 7 test trials + fixations before them
    assert len(df_trimmed) < len(df)
    assert cut_pos is not None
    assert cut_before_halfway is False  # 7/10 > 0.5
    assert proportion_blank == 0.3  # 3 blank out of 10 test trials


def test_preprocess_rt_tail_cutoff_trim_before_halfway():
    """Test trimming when cutoff occurs before halfway point."""
    # 20 test trials: first 5 have RTs, last 15 are blank
    df = pd.DataFrame({
        'trial_id': ['test_trial'] * 20,
        'rt': [100, 200, 300, 400, 500] + [-1] * 15
    })
    df_trimmed, cut_pos, cut_before_halfway, proportion_blank = preprocess_rt_tail_cutoff(
        df, subject_id='s01', task_name='flanker', last_n_test_trials=10
    )
    assert cut_pos is not None
    assert cut_before_halfway is True  # 5/20 < 0.5
    assert proportion_blank == 0.75  # 15 blank out of 20


def test_preprocess_rt_tail_cutoff_not_enough_trailing_blanks():
    """Test when last_n_test_trials requirement is not met."""
    # 10 test trials: last 5 are blank, but we require last 10 to be blank
    df = pd.DataFrame({
        'trial_id': ['test_trial'] * 10,
        'rt': [100, 200, 300, 400, 500, -1, -1, -1, -1, -1]
    })
    df_trimmed, cut_pos, cut_before_halfway, proportion_blank = preprocess_rt_tail_cutoff(
        df, subject_id='s01', task_name='flanker', last_n_test_trials=10
    )
    # Should not trim because not all last 10 are blank
    assert df_trimmed.equals(df)
    assert cut_pos is None
    assert proportion_blank == 0.5


def test_preprocess_rt_tail_cutoff_mixed_tail():
    """Test when tail has mixed valid and invalid responses (should not trim)."""
    df = pd.DataFrame({
        'trial_id': ['test_trial'] * 10,
        'rt': [100, 200, 300, 400, 500, -1, 600, -1, -1, -1]
    })
    df_trimmed, cut_pos, cut_before_halfway, proportion_blank = preprocess_rt_tail_cutoff(
        df, subject_id='s01', task_name='flanker', last_n_test_trials=5
    )
    # Should not trim because tail is not all -1
    assert df_trimmed.equals(df)
    assert cut_pos is None
    assert proportion_blank == 0.4


def test_preprocess_rt_tail_cutoff_no_valid_responses():
    """Test when there are no valid responses at all."""
    df = pd.DataFrame({
        'trial_id': ['test_trial'] * 10,
        'rt': [-1] * 10
    })
    df_trimmed, cut_pos, cut_before_halfway, proportion_blank = preprocess_rt_tail_cutoff(
        df, subject_id='s01', task_name='flanker'
    )
    assert df_trimmed.equals(df)
    assert cut_pos is None
    assert proportion_blank == 1.0


def test_preprocess_rt_tail_cutoff_no_test_trials():
    """Test when there are no test_trial rows."""
    df = pd.DataFrame({
        'trial_id': ['fixation'] * 10,
        'rt': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    })
    df_trimmed, cut_pos, cut_before_halfway, proportion_blank = preprocess_rt_tail_cutoff(
        df, subject_id='s01', task_name='flanker'
    )
    assert df_trimmed.equals(df)
    assert cut_pos is None
    assert proportion_blank == 0.0


def test_preprocess_rt_tail_cutoff_missing_columns():
    """Test when required columns are missing."""
    df = pd.DataFrame({
        'some_other_column': [1, 2, 3]
    })
    df_trimmed, cut_pos, cut_before_halfway, proportion_blank = preprocess_rt_tail_cutoff(
        df, subject_id='s01', task_name='flanker'
    )
    assert df_trimmed.equals(df)
    assert cut_pos is None
    assert proportion_blank == 0.0


def test_preprocess_rt_tail_cutoff_with_session():
    """Test trimming with session parameter."""
    df = pd.DataFrame({
        'trial_id': ['test_trial'] * 10,
        'rt': [100, 200, 300, 400, 500, 600, 700, -1, -1, -1]
    })
    df_trimmed, cut_pos, cut_before_halfway, proportion_blank = preprocess_rt_tail_cutoff(
        df, subject_id='s01', session='ses-01', task_name='flanker', last_n_test_trials=3
    )
    assert len(df_trimmed) < len(df)
    assert cut_pos is not None


def test_preprocess_rt_tail_cutoff_numeric_rt_with_nan():
    """Test handling of NaN values in RT column."""
    df = pd.DataFrame({
        'trial_id': ['test_trial'] * 10,
        'rt': [100, 200, np.nan, 400, 500, 600, 700, -1, -1, -1]
    })
    df_trimmed, cut_pos, cut_before_halfway, proportion_blank = preprocess_rt_tail_cutoff(
        df, subject_id='s01', task_name='flanker', last_n_test_trials=3
    )
    # NaN should be treated as -1 (non-response)
    assert cut_pos is not None


def test_get_bids_task_name_stop_signal():
    """Test BIDS task name conversion for stop signal tasks."""
    assert get_bids_task_name('stop_signal_with_directed_forgetting') == 'stopSignalWDirectedForgetting'
    assert get_bids_task_name('stop_signal_with_flanker') == 'stopSignalWFlanker'
    assert get_bids_task_name('stop_signal_single_task_network') == 'stopSignal'


def test_get_bids_task_name_go_nogo():
    """Test BIDS task name conversion for go/nogo tasks."""
    assert get_bids_task_name('go_nogo_single_task_network') == 'goNogo'
    assert get_bids_task_name('go_nogo_with_flanker') == 'goNogo'


def test_get_bids_task_name_shape_matching():
    """Test BIDS task name conversion for shape matching."""
    assert get_bids_task_name('shape_matching_single_task_network') == 'shapeMatching'
    assert get_bids_task_name('flanker_with_shape_matching') == 'shapeMatching'


def test_get_bids_task_name_cued_task_switching():
    """Test BIDS task name conversion for cued task switching."""
    assert get_bids_task_name('cued_task_switching_single_task_network') == 'cuedTS'
    assert get_bids_task_name('flanker_with_cued_task_switching') == 'cuedTS'


def test_get_bids_task_name_n_back():
    """Test BIDS task name conversion for n-back."""
    assert get_bids_task_name('n_back_single_task_network') == 'nBack'
    assert get_bids_task_name('n_back_with_flanker') == 'nBack'


def test_get_bids_task_name_flanker():
    """Test BIDS task name conversion for flanker."""
    assert get_bids_task_name('flanker_single_task_network') == 'flanker'
    # Note: flanker_with_cued_task_switching returns 'cuedTS' because cued_task_switching is checked first
    # This matches the actual function behavior where order matters
    assert get_bids_task_name('flanker_with_cued_task_switching') == 'cuedTS'


def test_get_bids_task_name_directed_forgetting():
    """Test BIDS task name conversion for directed forgetting."""
    assert get_bids_task_name('directed_forgetting_single_task_network') == 'directedForgetting'


def test_get_bids_task_name_spatial_task_switching():
    """Test BIDS task name conversion for spatial task switching."""
    assert get_bids_task_name('spatial_task_switching_single_task_network') == 'spatialTS'


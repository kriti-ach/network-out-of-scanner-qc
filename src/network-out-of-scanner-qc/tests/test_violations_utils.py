import pandas as pd
import numpy as np
from pathlib import Path
import pytest

from utils.violations_utils import (
    check_violation_conditions,
    find_difference,
    get_ssd,
    compute_violations,
    aggregate_violations,
    create_matrix_with_mean,
    create_violations_matrices,
    plot_violations,
)


def test_small_helpers():
    curr = {'stop_signal_condition': 'go', 'rt': 0.5}
    nxt = {'stop_signal_condition': 'stop', 'rt': 0.7, 'SS_delay': 0.2}
    assert check_violation_conditions(curr, nxt)
    assert find_difference(0.7, 0.5) == pytest.approx(0.2)
    assert get_ssd(nxt) == 0.2


def test_compute_violations_basic():
    # Sequence: go (rt=0.5) -> stop (rt=0.7) => violation row (difference 0.2)
    df = pd.DataFrame({
        'trial_id': ['test_trial', 'test_trial', 'test_trial'],
        'stop_signal_condition': ['go', 'stop', np.nan],
        'rt': [0.5, 0.7, 0.4],
        'SS_delay': [np.nan, 0.2, np.nan],
    })
    out = compute_violations('s01', df, 'stop_signal_with_flanker')
    assert len(out) == 1
    row = out.iloc[0]
    assert row['subject_id'] == 's01'
    assert row['task_name'] == 'stop_signal_with_flanker'
    assert row['ssd'] == 0.2
    assert row['difference'] == pytest.approx(0.2)
    assert row['violation'] == True  # go_rt < stop_rt

    # If SSD is NaN, row should be skipped
    df_nan_ssd = pd.DataFrame({
        'trial_id': ['test_trial', 'test_trial'],
        'stop_signal_condition': ['go', 'stop'],
        'rt': [0.6, 0.8],
        'SS_delay': [np.nan, np.nan],
    })
    out2 = compute_violations('s01', df_nan_ssd, 'stop_signal_with_flanker')
    assert len(out2) == 0


def test_aggregate_violations_and_matrices(tmp_path: Path):
    # Build a small violations dataframe to aggregate
    violations_df = pd.DataFrame({
        'subject_id': ['s01', 's01', 's02'],
        'task_name': ['stop_signal_with_flanker', 'stop_signal_with_flanker', 'stop_signal_with_flanker'],
        'ssd': [0.2, 0.3, 0.2],
        'difference': [0.1, 0.2, 0.4],
        'violation': [True, False, True],
    })

    agg = aggregate_violations(violations_df)
    # Expect grouped rows by subject x ssd
    assert set(agg.columns) == {
        'subject_id', 'task_name', 'ssd', 'difference_mean', 'proportion_violation', 'count_pairs'
    }
    # s01@0.2: one row, mean=0.1, prop=1.0, count=1
    row = agg[(agg['subject_id'] == 's01') & (agg['ssd'] == 0.2)].iloc[0]
    assert row['difference_mean'] == pytest.approx(0.1)
    assert row['proportion_violation'] == 1.0
    assert row['count_pairs'] == 1

    # Test matrix creation (CSV written with a mean row/col)
    count_matrix = agg.pivot(index='subject_id', columns='ssd', values='count_pairs')
    create_matrix_with_mean(count_matrix.copy(), tmp_path, 'matrix.csv')
    out_csv = tmp_path / 'matrix.csv'
    assert out_csv.exists()
    mat = pd.read_csv(out_csv, index_col=0)
    assert 'mean' in mat.index
    assert 'mean' in mat.columns



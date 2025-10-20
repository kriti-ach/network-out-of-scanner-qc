import pandas as pd
import os
from pathlib import Path
import re
import numpy as np

from utils.globals import (
    STOP_SUCCESS_ACC_LOW_THRESHOLD,
    STOP_SUCCESS_ACC_HIGH_THRESHOLD,
    GO_RT_THRESHOLD,
    GO_ACC_THRESHOLD_GO_NOGO,
    NOGO_ACC_THRESHOLD_GO_NOGO,
    GO_OMISSION_RATE_THRESHOLD,
    MISMATCH_THRESHOLD,
    MISMATCH_COMBINED_THRESHOLD,
    MATCH_THRESHOLD,
    MISMATCH_COMBINED_THRESHOLD,
    MATCH_COMBINED_THRESHOLD,
    ACC_THRESHOLD,
    OMISSION_RATE_THRESHOLD,
    NOGO_STOP_SUCCESS_MIN,
    SUMMARY_ROWS,
    GO_RT_THRESHOLD_DUAL_TASK
)
from utils.qc_utils import sort_subject_ids, is_dual_task

# Build maps by condition suffix to require same non-nback condition (e.g., flanker congruency, cuedTS state)
def suffix(col: str, prefix: str) -> str:
    # Everything after the prefix, preserves full condition detail
    idx = col.find(prefix)
    return col[idx + len(prefix):] if idx != -1 else col

def prefix(col: str, prefix: str) -> str:
    # Everything before the prefix, preserves full condition detail
    idx = col.find(prefix)
    return col[:idx] if idx != -1 else col

def check_exclusion_criteria(task_name, task_csv, exclusion_df):
        if 'stop_signal' in task_name:
            exclusion_df = check_stop_signal_exclusion_criteria(task_name, task_csv, exclusion_df)
        if 'go_nogo' in task_name:
            exclusion_df = check_go_nogo_exclusion_criteria(task_name, task_csv, exclusion_df)
        if 'n_back' in task_name:
            exclusion_df = check_n_back_exclusion_criteria(task_name, task_csv, exclusion_df)
        if ('flanker' in task_name or 'directed_forgetting' in task_name or 
        'shape_matching' in task_name or 'spatial_task_switching' in task_name or 
        'cued_task_switching' in task_name):
            exclusion_df = check_other_exclusion_criteria(task_name, task_csv, exclusion_df)
        return exclusion_df

def compare_to_threshold(metric_name, metric_value, threshold):
    """Check if a metric value violates the exclusion criteria."""
    return metric_value < threshold if 'low' in metric_name or 'acc' in metric_name else metric_value > threshold

def append_exclusion_row(exclusion_df, subject_id, metric_name, metric_value, threshold):
    """Append a new exclusion row to the exclusion dataframe, avoiding duplicates."""
    # Check if this subject+metric combination already exists
    if len(exclusion_df) > 0:
        existing = exclusion_df[
            (exclusion_df['subject_id'] == subject_id) & 
            (exclusion_df['metric'] == metric_name)
        ]
        if len(existing) > 0:
            return exclusion_df  # Skip duplicate
    
    # Append new row
    exclusion_df = pd.concat([
        exclusion_df,
        pd.DataFrame({
            'subject_id': [subject_id],
            'metric': [metric_name],
            'metric_value': [metric_value],
            'threshold': [threshold]
        })
    ], ignore_index=True)
    return exclusion_df

def check_stop_signal_exclusion_criteria(task_name, task_csv, exclusion_df):
    for index, row in task_csv.iterrows():
        #ignore the last 4 rows (summary rows)
        if index >= len(task_csv) - SUMMARY_ROWS:
            continue
        subject_id = row['subject_id']

        # Get actual column names for each metric type
        # Don't check nogo stop success
        stop_success_cols = [col for col in task_csv.columns if 'stop_success' in col]
        go_rt_cols = [col for col in task_csv.columns if 'go_rt' in col]
        go_acc_cols = [col for col in task_csv.columns if 'go_acc' in col]
        go_omission_rate_cols = [col for col in task_csv.columns if 'go_omission_rate' in col]
        stop_fail_rt_cols = [col for col in task_csv.columns if 'stop_fail_rt' in col]
        
        # Check stop_success specifically for low and high thresholds
        for col_name in stop_success_cols:
            value = row[col_name]
            if 'nogo' in col_name:
                if compare_to_threshold('stop_success_low', value, NOGO_STOP_SUCCESS_MIN):
                    exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, NOGO_STOP_SUCCESS_MIN)
            else:
                if compare_to_threshold('stop_success_low', value, STOP_SUCCESS_ACC_LOW_THRESHOLD):
                    exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, STOP_SUCCESS_ACC_LOW_THRESHOLD)
                if compare_to_threshold('stop_success_high', value, STOP_SUCCESS_ACC_HIGH_THRESHOLD):
                    exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, STOP_SUCCESS_ACC_HIGH_THRESHOLD)

        # Check go_rt columns
        for col_name in go_rt_cols:
            value = row[col_name]
            if compare_to_threshold('go_rt', value, GO_RT_THRESHOLD if not is_dual_task(task_name) else GO_RT_THRESHOLD_DUAL_TASK):
                exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, GO_RT_THRESHOLD)

        # Check if stop fail rt > go rt if the prefix is the same
        for col_name in stop_fail_rt_cols:
            for col_name_go in go_rt_cols:
                if prefix(col_name, 'stop_fail_rt') == prefix(col_name_go, 'go_rt'):
                    metric_name = f"{prefix(col_name, 'stop_fail_rt')}stop_fail_rt_greater_than_go_rt"
                    stop_fail_rt = row[col_name]
                    go_rt = row[col_name_go]
                    if stop_fail_rt > go_rt:
                        exclusion_df = append_exclusion_row(exclusion_df, subject_id, metric_name, stop_fail_rt, go_rt)
        # Check go_acc columns unless this is an N-back dual (N-back accuracy rules should own accuracy)
        if 'n_back' not in task_name:
            for col_name in go_acc_cols:
                value = row[col_name]
                if compare_to_threshold('go_acc', value, ACC_THRESHOLD):
                    exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, ACC_THRESHOLD)

        # Check go_omission_rate columns
        for col_name in go_omission_rate_cols:
            value = row[col_name]
            if compare_to_threshold('go_omission_rate', value, OMISSION_RATE_THRESHOLD):
                exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, OMISSION_RATE_THRESHOLD)

    #sort by subject_id
    exclusion_df = sort_subject_ids(exclusion_df)
    return exclusion_df

def check_go_nogo_exclusion_criteria(task_name, task_csv, exclusion_df):
    for index, row in task_csv.iterrows():
        #ignore the last 4 rows (summary rows)
        if index >= len(task_csv) - SUMMARY_ROWS:
            continue
        subject_id = row['subject_id']

        # Get actual column names for each metric type
        go_acc_cols = [col for col in task_csv.columns if 'go' in col and 'acc' in col and 'nogo' not in col]
        nogo_acc_cols = [col for col in task_csv.columns if 'nogo' in col and 'acc' in col]
        go_omission_rate_cols = [col for col in task_csv.columns if 'go' in col and 'omission_rate' in col and 'nogo' not in col]

        # If go accuracy < threshold AND nogo accuracy < threshold, then exclude
        # Only check when the prefix (before go_acc/nogo_acc) matches
        for col_name_go in go_acc_cols:
            for col_name_nogo in nogo_acc_cols:
                # Extract prefix before 'go_acc' and 'nogo_acc'
                go_prefix = col_name_go.replace('go_acc', '')
                nogo_prefix = col_name_nogo.replace('nogo_acc', '')
                
                # Only proceed if prefixes match
                if go_prefix == nogo_prefix:
                    go_acc_value = row[col_name_go]
                    nogo_acc_value = row[col_name_nogo]
                    # Check go accuracy threshold only if this is a go_nogo single task 
                    if task_name == "go_nogo_single_task_network":
                        if compare_to_threshold('go_acc', go_acc_value, GO_ACC_THRESHOLD_GO_NOGO):
                            exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name_go, go_acc_value, GO_ACC_THRESHOLD_GO_NOGO)
                    if compare_to_threshold('nogo_acc', nogo_acc_value, NOGO_ACC_THRESHOLD_GO_NOGO):
                        exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name_nogo, nogo_acc_value, NOGO_ACC_THRESHOLD_GO_NOGO)

        for col_name in go_omission_rate_cols:
            value = row[col_name]
            if compare_to_threshold('go_omission_rate', value, GO_OMISSION_RATE_THRESHOLD):
                exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, GO_OMISSION_RATE_THRESHOLD)
    #sort by subject_id
    if len(exclusion_df) != 0:
        exclusion_df = sort_subject_ids(exclusion_df)
    return exclusion_df

def check_n_back_exclusion_criteria(task_name, task_csv, exclusion_df):
    for index, row in task_csv.iterrows():
        #ignore the last 4 rows (summary rows)
        if index >= len(task_csv) - SUMMARY_ROWS:
            continue
        subject_id = row['subject_id']

        # Combined condition at the top: mismatch < threshold AND match < threshold
        exclusion_df = nback_flag_combined_accuracy(exclusion_df, subject_id, row, task_csv)

        for load in [1, 2, 3]:
            cols = nback_get_columns(task_csv, load)
            exclusion_df = nback_flag_independent_accuracy(exclusion_df, subject_id, row, load, cols)
            exclusion_df = nback_flag_omission_rates(exclusion_df, subject_id, row, load, cols)

    #sort by subject_id
    if len(exclusion_df) != 0:
        exclusion_df = sort_subject_ids(exclusion_df)
    return exclusion_df

def nback_flag_combined_accuracy(exclusion_df, subject_id, row, task_csv):
    """Flag N-back combined condition where both mismatch and match accuracies
    fall below their respective combined thresholds for a given load.

    Combined rule (by load): mismatch < 70% AND match < 55%.
    Applies across loads 1, 2, and 3 when columns exist.
    """
    for load in [1, 2, 3]:
        load_str = f"{load}.0back"
        mismatch_cols = [col for col in task_csv.columns if f'mismatch_{load_str}_' in col and 'acc' in col and 'nogo' not in col and 'stop_fail' not in col]
        match_cols = [col for col in task_csv.columns if f'match_{load_str}_' in col and 'acc' in col and 'mismatch' not in col and 'nogo' not in col and 'stop_fail' not in col]

        mismatch_map = {suffix(c, f"mismatch_{load_str}_"): c for c in mismatch_cols}
        match_map = {suffix(c, f"match_{load_str}_"): c for c in match_cols}
        common_suffixes = set(mismatch_map.keys()) & set(match_map.keys())

        for cond_suffix in common_suffixes:
            mismatch_col = mismatch_map[cond_suffix]
            match_col = match_map[cond_suffix]
            mismatch_val = row[mismatch_col]
            match_val = row[match_col]
            if pd.notna(mismatch_val) and pd.notna(match_val):
                if (mismatch_val < MISMATCH_COMBINED_THRESHOLD) and (match_val < MATCH_COMBINED_THRESHOLD):
                    exclusion_df = append_exclusion_row(
                        exclusion_df, subject_id, f'{mismatch_col}_combined', mismatch_val, MISMATCH_COMBINED_THRESHOLD
                    )
                    exclusion_df = append_exclusion_row(
                        exclusion_df, subject_id, f'{match_col}_combined', match_val, MATCH_COMBINED_THRESHOLD
                    )
    return exclusion_df

def nback_get_columns(task_csv, load):
    load_str = f"{load}.0back"
    match_acc = [col for col in task_csv.columns if f'match_{load_str}' in col and 'acc' in col and 'mismatch' not in col and 'nogo' not in col and 'stop_fail' not in col]
    mismatch_acc = [col for col in task_csv.columns if f'mismatch_{load_str}' in col and 'acc' in col and 'nogo' not in col and 'stop_fail' not in col]
    omiss = [col for col in task_csv.columns if f'{load_str}_omission_rate' in col]
    return {
        'match_acc': match_acc,
        'mismatch_acc': mismatch_acc,
        'omission_rate': omiss,
    }

def nback_flag_independent_accuracy(exclusion_df, subject_id, row, load, cols):
    # Flag mismatch accuracy below threshold (use full column name)
    for mismatch_col in cols['mismatch_acc']:
        val = row[mismatch_col]
        if compare_to_threshold(mismatch_col, val, MISMATCH_THRESHOLD):
            exclusion_df = append_exclusion_row(
                exclusion_df, subject_id, mismatch_col, val, MISMATCH_THRESHOLD
            )
    # Flag match accuracy below threshold (use full column name)
    for match_col in cols['match_acc']:
        val = row[match_col]
        if compare_to_threshold(match_col, val, MATCH_THRESHOLD):
            exclusion_df = append_exclusion_row(
                exclusion_df, subject_id, match_col, val, MATCH_THRESHOLD
            )
    return exclusion_df

def nback_flag_omission_rates(exclusion_df, subject_id, row, load, cols):
    for omiss_col in cols['omission_rate']:
        val = row[omiss_col]
        if compare_to_threshold(omiss_col, val, OMISSION_RATE_THRESHOLD):
            exclusion_df = append_exclusion_row(
                exclusion_df, subject_id, omiss_col, val, OMISSION_RATE_THRESHOLD
            )
    return exclusion_df

def check_other_exclusion_criteria(task_name, task_csv, exclusion_df):
    for index, row in task_csv.iterrows():
        if index >= len(task_csv) - SUMMARY_ROWS:
            continue
        subject_id = row['subject_id']
        
        # Get all accuracy and omission rate columns
        acc_cols = [col for col in task_csv.columns if 'acc' in col and 'nogo' not in col and 'stop_fail' not in col]
        omission_rate_cols = [col for col in task_csv.columns if 'omission_rate' in col]
        
        # If this task includes N-back, let N-back rules handle accuracy; still apply other tasks' omission rules
        if 'n_back' not in task_name:
            # Check accuracy columns only if this is not an N-back task
            for col_name in acc_cols:
                value = row[col_name]
                if compare_to_threshold(col_name, value, ACC_THRESHOLD):
                    exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, ACC_THRESHOLD)
        
        # Check omission rate columns (but exclude N-back specific omission rates)
        for col_name in omission_rate_cols:
            # Skip N-back specific omission rates (they are handled by N-back exclusion criteria)
            if 'back' not in col_name.lower():
                value = row[col_name]
                if compare_to_threshold(col_name, value, OMISSION_RATE_THRESHOLD):
                    exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, OMISSION_RATE_THRESHOLD)
    
    #sort by subject_id
    if len(exclusion_df) != 0:
        exclusion_df = sort_subject_ids(exclusion_df)
    return exclusion_df
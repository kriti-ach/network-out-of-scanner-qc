import pandas as pd
import os
from pathlib import Path
import re
import numpy as np

from utils.globals import (
    STOP_SUCCESS_ACC_LOW_THRESHOLD,
    STOP_SUCCESS_ACC_HIGH_THRESHOLD,
    GO_RT_THRESHOLD_OUT_OF_SCANNER,
    GO_RT_THRESHOLD_OUT_OF_SCANNER_DUAL_TASK,
    GO_RT_THRESHOLD_FMRI,
    GO_RT_THRESHOLD_FMRI_DUAL_TASK,
    GO_ACC_THRESHOLD_GO_NOGO,
    NOGO_ACC_THRESHOLD_GO_NOGO,
    GO_OMISSION_RATE_THRESHOLD,
    MISMATCH_THRESHOLD,
    MISMATCH_COMBINED_THRESHOLD,
    MATCH_THRESHOLD,
    MATCH_COMBINED_THRESHOLD,
    ACC_THRESHOLD,
    OMISSION_RATE_THRESHOLD,
    NOGO_STOP_SUCCESS_MIN,
    SUMMARY_ROWS,
    NBACK_1BACK_MATCH_ACC_COMBINED_THRESHOLD_1,
    NBACK_1BACK_MISMATCH_ACC_COMBINED_THRESHOLD_1,
    NBACK_1BACK_MATCH_ACC_COMBINED_THRESHOLD_2,
    NBACK_1BACK_MISMATCH_ACC_COMBINED_THRESHOLD_2,
    NBACK_2BACK_MATCH_ACC_COMBINED_THRESHOLD_1,
    NBACK_2BACK_MISMATCH_ACC_COMBINED_THRESHOLD_1,
    NBACK_2BACK_MATCH_ACC_COMBINED_THRESHOLD_2,
    NBACK_2BACK_MISMATCH_ACC_COMBINED_THRESHOLD_2,
    GONOGO_GO_ACC_THRESHOLD_1,
    GONOGO_NOGO_ACC_THRESHOLD_1,
    GONOGO_GO_ACC_THRESHOLD_2,
    GONOGO_NOGO_ACC_THRESHOLD_2
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

def append_exclusion_row(exclusion_df, subject_id, metric_name, metric_value, threshold, session=None):
    """Append a new exclusion row to the exclusion dataframe, avoiding duplicates."""
    # Check if this subject+metric+session combination already exists
    if len(exclusion_df) > 0:
        # Build duplicate check condition
        duplicate_condition = (exclusion_df['subject_id'] == subject_id) & (exclusion_df['metric'] == metric_name)
        
        # If session is provided, also check session (for in-scanner data)
        if session is not None:
            # Ensure session column exists
            if 'session' in exclusion_df.columns:
                duplicate_condition = duplicate_condition & (exclusion_df['session'] == session)
        
        existing = exclusion_df[duplicate_condition]
        if len(existing) > 0:
            return exclusion_df  # Skip duplicate
    
    # Ensure session column exists and is in the right position (after subject_id)
    if session is not None and 'session' not in exclusion_df.columns:
        # Insert session column right after subject_id
        cols = list(exclusion_df.columns)
        subj_idx = cols.index('subject_id') if 'subject_id' in cols else 0
        cols.insert(subj_idx + 1, 'session')
        exclusion_df = exclusion_df.reindex(columns=cols)
        exclusion_df['session'] = pd.Series(dtype=str)
    
    # Append new row
    row_dict = {
        'subject_id': [subject_id],
        'metric': [metric_name],
        'metric_value': [metric_value],
        'threshold': [threshold]
    }
    if session is not None:
        row_dict['session'] = [session]
    
    # Ensure row_dict columns match exclusion_df columns (with session in correct position)
    new_row = pd.DataFrame(row_dict)
    for col in exclusion_df.columns:
        if col not in new_row.columns:
            new_row[col] = None
    
    # Reorder columns to match exclusion_df
    new_row = new_row[exclusion_df.columns]
    exclusion_df = pd.concat([exclusion_df, new_row], ignore_index=True)
    return exclusion_df

def check_stop_signal_exclusion_criteria(task_name, task_csv, exclusion_df):
    # Detect if this is fMRI mode (has session column)
    is_fmri = 'session' in task_csv.columns
    is_stop_dual = is_dual_task(task_name) and 'stop_signal' in task_name
    
    for index, row in task_csv.iterrows():
        #ignore the last 4 rows (summary rows)
        if index >= len(task_csv) - SUMMARY_ROWS:
            continue
        subject_id = row['subject_id']
        session = row['session'] if 'session' in row.index else None

        # For in-scanner stop signal dual tasks: use overall metrics for exclusion
        if is_fmri and is_stop_dual:
            # Check overall_go_acc
            if 'overall_go_acc' in task_csv.columns:
                overall_go_acc = row['overall_go_acc']
                if pd.notna(overall_go_acc) and compare_to_threshold('overall_go_acc', overall_go_acc, ACC_THRESHOLD):
                    exclusion_df = append_exclusion_row(exclusion_df, subject_id, 'overall_go_acc', overall_go_acc, ACC_THRESHOLD, session)
            
            # Check overall_go_rt
            if 'overall_go_rt' in task_csv.columns:
                overall_go_rt = row['overall_go_rt']
                if pd.notna(overall_go_rt) and compare_to_threshold('overall_go_rt', overall_go_rt, GO_RT_THRESHOLD_FMRI):
                    exclusion_df = append_exclusion_row(exclusion_df, subject_id, 'overall_go_rt', overall_go_rt, GO_RT_THRESHOLD_FMRI, session)
            
            # Check overall_stop_success (low threshold)
            if 'overall_stop_success' in task_csv.columns:
                overall_stop_success = row['overall_stop_success']
                if pd.notna(overall_stop_success) and compare_to_threshold('stop_success_low', overall_stop_success, STOP_SUCCESS_ACC_LOW_THRESHOLD):
                    exclusion_df = append_exclusion_row(exclusion_df, subject_id, 'overall_stop_success', overall_stop_success, STOP_SUCCESS_ACC_LOW_THRESHOLD, session)
            
            # Check overall_stop_success (high threshold)
            if 'overall_stop_success' in task_csv.columns:
                overall_stop_success = row['overall_stop_success']
                if pd.notna(overall_stop_success) and compare_to_threshold('stop_success_high', overall_stop_success, STOP_SUCCESS_ACC_HIGH_THRESHOLD):
                    exclusion_df = append_exclusion_row(exclusion_df, subject_id, 'overall_stop_success', overall_stop_success, STOP_SUCCESS_ACC_HIGH_THRESHOLD, session)
            
            # Condition-specific criteria will be moved to flags (handled in main.py)
        else:
            # For out-of-scanner or single tasks: use existing logic
            # Get actual column names for each metric type
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
                        exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, NOGO_STOP_SUCCESS_MIN, session)
                else:
                    if compare_to_threshold('stop_success_low', value, STOP_SUCCESS_ACC_LOW_THRESHOLD):
                        exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, STOP_SUCCESS_ACC_LOW_THRESHOLD, session)
                    if compare_to_threshold('stop_success_high', value, STOP_SUCCESS_ACC_HIGH_THRESHOLD):
                        exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, STOP_SUCCESS_ACC_HIGH_THRESHOLD, session)

            # Check go_rt columns - use fMRI threshold if in fMRI mode
            for col_name in go_rt_cols:
                value = row[col_name]
                if is_fmri:
                    rt_threshold = GO_RT_THRESHOLD_FMRI if not is_stop_dual else GO_RT_THRESHOLD_FMRI_DUAL_TASK
                else:
                    rt_threshold = GO_RT_THRESHOLD_OUT_OF_SCANNER if not is_stop_dual else GO_RT_THRESHOLD_OUT_OF_SCANNER_DUAL_TASK
                if compare_to_threshold('go_rt', value, rt_threshold):
                    exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, rt_threshold, session)

            # Check if stop fail rt > go rt if the prefix is the same
            for col_name in stop_fail_rt_cols:
                for col_name_go in go_rt_cols:
                    if prefix(col_name, 'stop_fail_rt') == prefix(col_name_go, 'go_rt'):
                        metric_name = f"{prefix(col_name, 'stop_fail_rt')}stop_fail_rt_greater_than_go_rt"
                        stop_fail_rt = row[col_name]
                        go_rt = row[col_name_go]
                        if stop_fail_rt > go_rt:
                            exclusion_df = append_exclusion_row(exclusion_df, subject_id, metric_name, stop_fail_rt, go_rt, session)
            
            # Check go_acc columns unless this is an N-back dual (N-back accuracy rules should own accuracy)
            if 'n_back' not in task_name:
                for col_name in go_acc_cols:
                    value = row[col_name]
                    if compare_to_threshold('go_acc', value, ACC_THRESHOLD):
                        exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, ACC_THRESHOLD, session)

            # Check go_omission_rate columns - skip exclusion for fMRI (will be flagged instead)
            if not is_fmri:
                for col_name in go_omission_rate_cols:
                    value = row[col_name]
                    if compare_to_threshold('go_omission_rate', value, OMISSION_RATE_THRESHOLD):
                        exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, OMISSION_RATE_THRESHOLD, session)

    #sort by subject_id
    exclusion_df = sort_subject_ids(exclusion_df)
    return exclusion_df

def check_go_nogo_exclusion_criteria(task_name, task_csv, exclusion_df):
    # Detect if this is fMRI mode (has session column)
    is_fmri = 'session' in task_csv.columns
    
    for index, row in task_csv.iterrows():
        #ignore the last 4 rows (summary rows)
        if index >= len(task_csv) - SUMMARY_ROWS:
            continue
        subject_id = row['subject_id']
        session = row['session'] if 'session' in row.index else None

        # Get actual column names for each metric type
        go_acc_cols = [col for col in task_csv.columns if 'go' in col and 'acc' in col and 'nogo' not in col]
        nogo_acc_cols = [col for col in task_csv.columns if 'nogo' in col and 'acc' in col]
        go_omission_rate_cols = [col for col in task_csv.columns if 'go' in col and 'omission_rate' in col and 'nogo' not in col]

        if is_fmri:
            # For fMRI: use new exclusion criteria
            # Check when the prefix (before go_acc/nogo_acc) matches
            for col_name_go in go_acc_cols:
                for col_name_nogo in nogo_acc_cols:
                    # Extract prefix before 'go_acc' and 'nogo_acc'
                    go_prefix = col_name_go.replace('go_acc', '')
                    nogo_prefix = col_name_nogo.replace('nogo_acc', '')
                    
                    # Only proceed if prefixes match
                    if go_prefix == nogo_prefix:
                        go_acc_value = row[col_name_go]
                        nogo_acc_value = row[col_name_nogo]
                        
                        # New exclusion criteria: (go < 0.8 or nogo < 0.2) AND (go < 0.55 or nogo < 0.55)
                        if pd.notna(go_acc_value) and pd.notna(nogo_acc_value):
                            exclude_rule1 = (go_acc_value <= GONOGO_GO_ACC_THRESHOLD_1) or (nogo_acc_value <= GONOGO_NOGO_ACC_THRESHOLD_1)
                            exclude_rule2 = (go_acc_value <= GONOGO_GO_ACC_THRESHOLD_2) or (nogo_acc_value <= GONOGO_NOGO_ACC_THRESHOLD_2)
                            
                            if exclude_rule1 and exclude_rule2:
                                # Exclude based on whichever rule was triggered
                                if exclude_rule1:
                                    exclusion_df = append_exclusion_row(exclusion_df, subject_id, f'{col_name_go}_fmri_rule1', go_acc_value, GONOGO_GO_ACC_THRESHOLD_1, session)
                                    exclusion_df = append_exclusion_row(exclusion_df, subject_id, f'{col_name_nogo}_fmri_rule1', nogo_acc_value, GONOGO_NOGO_ACC_THRESHOLD_1, session)
                                if exclude_rule2:
                                    exclusion_df = append_exclusion_row(exclusion_df, subject_id, f'{col_name_go}_fmri_rule2', go_acc_value, GONOGO_GO_ACC_THRESHOLD_2, session)
                                    exclusion_df = append_exclusion_row(exclusion_df, subject_id, f'{col_name_nogo}_fmri_rule2', nogo_acc_value, GONOGO_NOGO_ACC_THRESHOLD_2, session)
        else:
            # For out-of-scanner: use existing criteria
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
                                exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name_go, go_acc_value, GO_ACC_THRESHOLD_GO_NOGO, session)
                            if compare_to_threshold('nogo_acc', nogo_acc_value, NOGO_ACC_THRESHOLD_GO_NOGO):
                                exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name_nogo, nogo_acc_value, NOGO_ACC_THRESHOLD_GO_NOGO, session)

            # Check go_omission_rate columns
            for col_name in go_omission_rate_cols:
                value = row[col_name]
                if compare_to_threshold('go_omission_rate', value, GO_OMISSION_RATE_THRESHOLD):
                    exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, GO_OMISSION_RATE_THRESHOLD, session)
    #sort by subject_id
    if len(exclusion_df) != 0:
        exclusion_df = sort_subject_ids(exclusion_df)
    return exclusion_df

def check_n_back_exclusion_criteria(task_name, task_csv, exclusion_df):
    # Detect if this is fMRI mode (has session column)
    is_fmri = 'session' in task_csv.columns
    is_nback_dual = is_dual_task(task_name) and 'n_back' in task_name
    
    for index, row in task_csv.iterrows():
        #ignore the last 4 rows (summary rows)
        if index >= len(task_csv) - SUMMARY_ROWS:
            continue
        subject_id = row['subject_id']
        session = row['session'] if 'session' in row.index else None
        for load in [1, 2, 3]:
            cols = nback_get_columns(task_csv, load)
            if is_fmri:
                # For fMRI: move independent accuracy checks to flagged (handled in main.py)
                # Only check new exclusion criteria
                if is_nback_dual:
                    # For dual tasks: use overall match/mismatch metrics
                    exclusion_df = nback_check_fmri_exclusion_criteria_dual(exclusion_df, subject_id, row, load, task_csv, session)
                else:
                    # For single tasks: use condition-specific criteria
                    exclusion_df = nback_check_fmri_exclusion_criteria(exclusion_df, subject_id, row, load, task_csv, session)
            else:
                # For out-of-scanner: use existing criteria
                exclusion_df = nback_flag_independent_accuracy(exclusion_df, subject_id, row, load, cols, session)
                exclusion_df = nback_flag_omission_rates(exclusion_df, subject_id, row, load, cols, session)
        # Check combined accuracy thresholds (once per row, not per load)
        if not is_fmri:
            # For out-of-scanner: use existing combined criteria
            exclusion_df = nback_flag_combined_accuracy(exclusion_df, subject_id, row, task_csv, session)
        # For fMRI: combined accuracy goes to flagged (handled in main.py)

    #sort by subject_id
    if len(exclusion_df) != 0:
        exclusion_df = sort_subject_ids(exclusion_df)
    return exclusion_df

def nback_check_fmri_exclusion_criteria(exclusion_df, subject_id, row, load, task_csv, session=None):
    """
    Check new fMRI exclusion criteria for n-back single tasks.
    
    For 1-back and 2-back only:
    - Rule 1: match < 0.2 AND mismatch < 0.75
    - Rule 2: match < 0.5 AND mismatch < 0.5
    
    Exclude if either rule is met.
    """
    if load not in [1, 2]:
        return exclusion_df
    
    load_str = f"{load}.0back"
    mismatch_cols = [col for col in task_csv.columns if f'mismatch_{load_str}' in col and 'acc' in col and 'nogo' not in col and 'stop_fail' not in col and 'overall' not in col]
    match_cols = [col for col in task_csv.columns if f'match_{load_str}' in col and 'acc' in col and 'mismatch' not in col and 'nogo' not in col and 'stop_fail' not in col and 'overall' not in col]
    
    mismatch_map = {}
    for col in mismatch_cols:
        prefix_with_underscore = f"mismatch_{load_str}_"
        if prefix_with_underscore in col:
            cond_suffix = suffix(col, prefix_with_underscore)
        else:
            prefix_no_underscore = f"mismatch_{load_str}"
            cond_suffix = suffix(col, prefix_no_underscore)
        mismatch_map[cond_suffix] = col
    
    match_map = {}
    for col in match_cols:
        prefix_with_underscore = f"match_{load_str}_"
        if prefix_with_underscore in col:
            cond_suffix = suffix(col, prefix_with_underscore)
        else:
            prefix_no_underscore = f"match_{load_str}"
            cond_suffix = suffix(col, prefix_no_underscore)
        match_map[cond_suffix] = col
    
    common_suffixes = set(mismatch_map.keys()) & set(match_map.keys())
    
    for cond_suffix in common_suffixes:
        mismatch_col = mismatch_map[cond_suffix]
        match_col = match_map[cond_suffix]
        mismatch_val = row[mismatch_col]
        match_val = row[match_col]
        
        if pd.notna(mismatch_val) and pd.notna(match_val):
            # Get thresholds based on load
            if load == 1:
                match_thresh_1 = NBACK_1BACK_MATCH_ACC_COMBINED_THRESHOLD_1
                mismatch_thresh_1 = NBACK_1BACK_MISMATCH_ACC_COMBINED_THRESHOLD_1
                match_thresh_2 = NBACK_1BACK_MATCH_ACC_COMBINED_THRESHOLD_2
                mismatch_thresh_2 = NBACK_1BACK_MISMATCH_ACC_COMBINED_THRESHOLD_2
            else:  # load == 2
                match_thresh_1 = NBACK_2BACK_MATCH_ACC_COMBINED_THRESHOLD_1
                mismatch_thresh_1 = NBACK_2BACK_MISMATCH_ACC_COMBINED_THRESHOLD_1
                match_thresh_2 = NBACK_2BACK_MATCH_ACC_COMBINED_THRESHOLD_2
                mismatch_thresh_2 = NBACK_2BACK_MISMATCH_ACC_COMBINED_THRESHOLD_2
            
            # Check both rules: (match < thresh1 AND mismatch < thresh1) OR (match < thresh2 AND mismatch < thresh2)
            exclude_rule1 = (match_val <= match_thresh_1) and (mismatch_val <= mismatch_thresh_1)
            exclude_rule2 = (match_val <= match_thresh_2) and (mismatch_val <= mismatch_thresh_2)
            
            if exclude_rule1 or exclude_rule2:
                # Exclude based on whichever rule was triggered
                if exclude_rule1:
                    exclusion_df = append_exclusion_row(
                        exclusion_df, subject_id, f'{mismatch_col}_fmri_rule1', mismatch_val, mismatch_thresh_1, session
                    )
                    exclusion_df = append_exclusion_row(
                        exclusion_df, subject_id, f'{match_col}_fmri_rule1', match_val, match_thresh_1, session
                    )
                if exclude_rule2:
                    exclusion_df = append_exclusion_row(
                        exclusion_df, subject_id, f'{mismatch_col}_fmri_rule2', mismatch_val, mismatch_thresh_2, session
                    )
                    exclusion_df = append_exclusion_row(
                        exclusion_df, subject_id, f'{match_col}_fmri_rule2', match_val, match_thresh_2, session
                    )
    
    return exclusion_df

def nback_check_fmri_exclusion_criteria_dual(exclusion_df, subject_id, row, load, task_csv, session=None):
    """
    Check new fMRI exclusion criteria for n-back dual tasks using overall match/mismatch metrics.
    
    For 1-back and 2-back only:
    - Rule 1: overall_match < 0.2 AND overall_mismatch < 0.75
    - Rule 2: overall_match < 0.5 AND overall_mismatch < 0.5
    
    Exclude if either rule is met.
    """
    if load not in [1, 2]:
        return exclusion_df
    
    load_str = f"{load}.0back"
    overall_match_col = f'overall_match_{load_str}_acc'
    overall_mismatch_col = f'overall_mismatch_{load_str}_acc'
    
    # Check if overall columns exist
    if overall_match_col not in task_csv.columns or overall_mismatch_col not in task_csv.columns:
        return exclusion_df
    
    match_val = row[overall_match_col]
    mismatch_val = row[overall_mismatch_col]
    
    if pd.notna(mismatch_val) and pd.notna(match_val):
        # Get thresholds based on load
        if load == 1:
            match_thresh_1 = NBACK_1BACK_MATCH_ACC_COMBINED_THRESHOLD_1
            mismatch_thresh_1 = NBACK_1BACK_MISMATCH_ACC_COMBINED_THRESHOLD_1
            match_thresh_2 = NBACK_1BACK_MATCH_ACC_COMBINED_THRESHOLD_2
            mismatch_thresh_2 = NBACK_1BACK_MISMATCH_ACC_COMBINED_THRESHOLD_2
        else:  # load == 2
            match_thresh_1 = NBACK_2BACK_MATCH_ACC_COMBINED_THRESHOLD_1
            mismatch_thresh_1 = NBACK_2BACK_MISMATCH_ACC_COMBINED_THRESHOLD_1
            match_thresh_2 = NBACK_2BACK_MATCH_ACC_COMBINED_THRESHOLD_2
            mismatch_thresh_2 = NBACK_2BACK_MISMATCH_ACC_COMBINED_THRESHOLD_2
        
        # Check both rules: (match < thresh1 AND mismatch < thresh1) OR (match < thresh2 AND mismatch < thresh2)
        exclude_rule1 = (match_val <= match_thresh_1) and (mismatch_val <= mismatch_thresh_1)
        exclude_rule2 = (match_val <= match_thresh_2) and (mismatch_val <= mismatch_thresh_2)
        
        if exclude_rule1 or exclude_rule2:
            # Exclude based on whichever rule was triggered
            if exclude_rule1:
                exclusion_df = append_exclusion_row(
                    exclusion_df, subject_id, f'{overall_mismatch_col}_fmri_rule1', mismatch_val, mismatch_thresh_1, session
                )
                exclusion_df = append_exclusion_row(
                    exclusion_df, subject_id, f'{overall_match_col}_fmri_rule1', match_val, match_thresh_1, session
                )
            if exclude_rule2:
                exclusion_df = append_exclusion_row(
                    exclusion_df, subject_id, f'{overall_mismatch_col}_fmri_rule2', mismatch_val, mismatch_thresh_2, session
                )
                exclusion_df = append_exclusion_row(
                    exclusion_df, subject_id, f'{overall_match_col}_fmri_rule2', match_val, match_thresh_2, session
                )
    
    return exclusion_df

def nback_flag_combined_accuracy(exclusion_df, subject_id, row, task_csv, session=None):
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
                        exclusion_df, subject_id, f'{mismatch_col}_combined', mismatch_val, MISMATCH_COMBINED_THRESHOLD, session
                    )
                    exclusion_df = append_exclusion_row(
                        exclusion_df, subject_id, f'{match_col}_combined', match_val, MATCH_COMBINED_THRESHOLD, session
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

def nback_flag_independent_accuracy(exclusion_df, subject_id, row, load, cols, session=None):
    # Flag mismatch accuracy below threshold (use full column name)
    for mismatch_col in cols['mismatch_acc']:
        val = row[mismatch_col]
        if compare_to_threshold(mismatch_col, val, MISMATCH_THRESHOLD):
            exclusion_df = append_exclusion_row(
                exclusion_df, subject_id, mismatch_col, val, MISMATCH_THRESHOLD, session
            )
    # Flag match accuracy below threshold (use full column name)
    for match_col in cols['match_acc']:
        val = row[match_col]
        if compare_to_threshold(match_col, val, MATCH_THRESHOLD):
            exclusion_df = append_exclusion_row(
                exclusion_df, subject_id, match_col, val, MATCH_THRESHOLD, session
            )
    return exclusion_df

def nback_flag_omission_rates(exclusion_df, subject_id, row, load, cols, session=None):
    for omiss_col in cols['omission_rate']:
        val = row[omiss_col]
        if compare_to_threshold(omiss_col, val, OMISSION_RATE_THRESHOLD):
            exclusion_df = append_exclusion_row(
                exclusion_df, subject_id, omiss_col, val, OMISSION_RATE_THRESHOLD, session
            )
    return exclusion_df

def check_other_exclusion_criteria(task_name, task_csv, exclusion_df):
    # Detect if this is fMRI mode (has session column)
    is_fmri = 'session' in task_csv.columns
    
    for index, row in task_csv.iterrows():
        if index >= len(task_csv) - SUMMARY_ROWS:
            continue
        subject_id = row['subject_id']
        session = row['session'] if 'session' in row.index else None
        
        # Get all accuracy and omission rate columns
        acc_cols = [col for col in task_csv.columns if 'acc' in col and 'nogo' not in col and 'stop_fail' not in col and col != 'overall_acc']
        omission_rate_cols = [col for col in task_csv.columns if 'omission_rate' in col]
        
        # If this task includes N-back, let N-back rules handle accuracy; still apply other tasks' omission rules
        if 'n_back' not in task_name:
            if is_fmri:
                # For fMRI: only check overall accuracy, move condition accuracies to flagged
                if 'overall_acc' in task_csv.columns:
                    overall_acc = row['overall_acc']
                    if pd.notna(overall_acc) and compare_to_threshold('overall_acc', overall_acc, ACC_THRESHOLD):
                        exclusion_df = append_exclusion_row(exclusion_df, subject_id, 'overall_acc', overall_acc, ACC_THRESHOLD, session)
                # Condition accuracies will be moved to flagged data (handled in main.py)
            else:
                # For out-of-scanner: check individual condition accuracies
                for col_name in acc_cols:
                    value = row[col_name]
                    if compare_to_threshold(col_name, value, ACC_THRESHOLD):
                        exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, ACC_THRESHOLD, session)
        else:
            # For n-back tasks in fMRI: exclude based on condition-specific accuracy (match/mismatch), not overall_acc
            # This is already handled by check_n_back_exclusion_criteria, so we don't need to do anything here
            pass
        
        # Check omission rate columns (but exclude N-back specific omission rates)
        # Skip exclusion for fMRI (will be flagged instead)
        if not is_fmri:
            for col_name in omission_rate_cols:
                # Skip N-back specific omission rates (they are handled by N-back exclusion criteria)
                if 'back' not in col_name.lower():
                    value = row[col_name]
                    if compare_to_threshold(col_name, value, OMISSION_RATE_THRESHOLD):
                        exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, OMISSION_RATE_THRESHOLD, session)
    
    #sort by subject_id
    if len(exclusion_df) != 0:
        exclusion_df = sort_subject_ids(exclusion_df)
    return exclusion_df

def remove_some_flags_for_exclusion(task_name, exclusion_df):
    # Convert metric column to string to handle mixed types
    exclusion_df['metric'] = exclusion_df['metric'].astype(str)
    
    # Remove rows where metric contains 'stop_fail_rt_greater_than_go_rt'
    exclusion_df = exclusion_df[~exclusion_df['metric'].str.contains('stop_fail_rt_greater_than_go_rt', na=False)]
    
    # Remove rows where metric contains '3.0back'
    exclusion_df = exclusion_df[~exclusion_df['metric'].str.contains('3.0back', na=False)]

    #Remove rows in stop+nback where metric contains stop_success but not collapsed
    if 'stop_signal' in task_name and 'n_back' in task_name:
        # Keep rows that either don't contain 'stop_success' OR contain 'collapsed'
        exclusion_df = exclusion_df[(~exclusion_df['metric'].str.contains('stop_success', na=False)) | (exclusion_df['metric'].str.contains('collapsed', na=False))]
    
    return exclusion_df

def create_combined_exclusions_csv(tasks, exclusions_output_path):
    """
    Create combined and summarized exclusions CSVs from all individual task exclusion files.
    
    Args:
        tasks (list): List of task names
        exclusions_output_path (Path): Path to the exclusions output folder
        
    Returns:
        None: Saves two CSVs:
            - all_exclusions.csv: All exclusion rows with all details
            - summarized_exclusions.csv: One row per subject-session-task combination
    """
    all_exclusions = []
    for task in tasks:
        exclusion_file = exclusions_output_path / f"excluded_data_{task}.csv"
        if exclusion_file.exists():
            try:
                task_exclusions = pd.read_csv(exclusion_file)
                if len(task_exclusions) > 0:
                    # Add task_name column
                    task_exclusions['task_name'] = task
                    # Reorder columns: subject_id, session (if exists), task_name, then rest
                    cols = list(task_exclusions.columns)
                    if 'session' in cols:
                        # For in-scanner: subject_id, session, task_name, then rest
                        cols.remove('subject_id')
                        cols.remove('session')
                        cols.remove('task_name')
                        task_exclusions = task_exclusions[['subject_id', 'session', 'task_name'] + cols]
                    else:
                        # For out-of-scanner: subject_id, task_name, then rest
                        cols.remove('subject_id')
                        cols.remove('task_name')
                        task_exclusions = task_exclusions[['subject_id', 'task_name'] + cols]
                    all_exclusions.append(task_exclusions)
            except Exception as e:
                print(f"Error reading exclusion file for {task}: {str(e)}")
    
    if len(all_exclusions) > 0:
        combined_exclusions = pd.concat(all_exclusions, ignore_index=True)
        # Sort by subject_id (and session if present)
        if 'session' in combined_exclusions.columns:
            combined_exclusions = combined_exclusions.sort_values(['subject_id', 'session', 'task_name']).reset_index(drop=True)
        else:
            combined_exclusions = combined_exclusions.sort_values(['subject_id', 'task_name']).reset_index(drop=True)
        
        # Save all_exclusions.csv with all details
        combined_exclusions.to_csv(exclusions_output_path / 'all_exclusions.csv', index=False)
        
        # Create summarized_exclusions.csv: one row per subject-session-task combination
        if 'session' in combined_exclusions.columns:
            # For in-scanner: group by subject_id, session, and task_name
            summarized = combined_exclusions[['subject_id', 'session', 'task_name']].drop_duplicates()
            summarized = summarized.sort_values(['subject_id', 'session', 'task_name']).reset_index(drop=True)
        else:
            # For out-of-scanner: group by subject_id and task_name
            summarized = combined_exclusions[['subject_id', 'task_name']].drop_duplicates()
            summarized = summarized.sort_values(['subject_id', 'task_name']).reset_index(drop=True)
        
        summarized.to_csv(exclusions_output_path / 'summarized_exclusions.csv', index=False)


def flag_fmri_condition_metrics(task_name, task_csv):
    """
    Flag condition accuracies and omission rates for fMRI tasks.
    These are moved to flagged data instead of exclusion data.
    
    Args:
        task_name (str): Name of the task
        task_csv (pd.DataFrame): QC CSV DataFrame with session column
        
    Returns:
        tuple: (condition_acc_flags_df, omission_rate_flags_df) - DataFrames with flagged metrics
    """
    condition_acc_flags = pd.DataFrame({'subject_id': [], 'metric': [], 'metric_value': [], 'threshold': []})
    omission_rate_flags = pd.DataFrame({'subject_id': [], 'metric': [], 'metric_value': [], 'threshold': []})
    if 'session' not in condition_acc_flags.columns:
        condition_acc_flags.insert(1, 'session', pd.Series(dtype=str))
        omission_rate_flags.insert(1, 'session', pd.Series(dtype=str))
    
    # Get all condition accuracy columns (exclude overall_acc)
    acc_cols = [col for col in task_csv.columns if col.endswith('_acc') and col != 'overall_acc' and 'nogo' not in col and 'stop_fail' not in col]
    
    # For n-back tasks: flag match/mismatch accuracy using their specific thresholds
    nback_match_cols = []
    nback_mismatch_cols = []
    if 'n_back' in task_name:
        nback_match_cols = [col for col in task_csv.columns if 'match_' in col and 'acc' in col and 'mismatch' not in col and 'nogo' not in col and 'stop_fail' not in col]
        nback_mismatch_cols = [col for col in task_csv.columns if 'mismatch_' in col and 'acc' in col and 'nogo' not in col and 'stop_fail' not in col]
        # Remove match/mismatch from general acc_cols
        acc_cols = [col for col in acc_cols if 'match_' not in col and 'mismatch_' not in col]
    
    # For go_nogo tasks: flag go_acc and nogo_acc using their specific thresholds
    go_nogo_go_acc_cols = []
    go_nogo_nogo_acc_cols = []
    if 'go_nogo' in task_name:
        go_nogo_go_acc_cols = [col for col in task_csv.columns if 'go' in col and 'acc' in col and 'nogo' not in col]
        go_nogo_nogo_acc_cols = [col for col in task_csv.columns if 'nogo' in col and 'acc' in col]
        # Remove go/nogo from general acc_cols
        acc_cols = [col for col in acc_cols if 'go_acc' not in col]
    
    # Get all omission rate columns
    omission_rate_cols = [col for col in task_csv.columns if 'omission_rate' in col]
    
    for idx, (index, row) in enumerate(task_csv.iterrows()):
        if idx >= len(task_csv) - 4:  # Skip summary rows
            continue
        subject_id = row['subject_id']
        session = row['session'] if 'session' in row.index else None
        
        # Flag n-back match accuracy
        for col_name in nback_match_cols:
            value = row[col_name]
            if pd.notna(value) and compare_to_threshold(col_name, value, MATCH_THRESHOLD):
                condition_acc_flags = append_exclusion_row(condition_acc_flags, subject_id, col_name, value, MATCH_THRESHOLD, session)
        
        # Flag n-back mismatch accuracy
        for col_name in nback_mismatch_cols:
            value = row[col_name]
            if pd.notna(value) and compare_to_threshold(col_name, value, MISMATCH_THRESHOLD):
                condition_acc_flags = append_exclusion_row(condition_acc_flags, subject_id, col_name, value, MISMATCH_THRESHOLD, session)
        
        # Flag n-back combined accuracy (using existing combined thresholds)
        if 'n_back' in task_name:
            temp_exclusion_df = pd.DataFrame({'subject_id': [], 'metric': [], 'metric_value': [], 'threshold': []})
            if 'session' not in temp_exclusion_df.columns:
                temp_exclusion_df.insert(1, 'session', pd.Series(dtype=str))
            temp_exclusion_df = nback_flag_combined_accuracy(temp_exclusion_df, subject_id, row, task_csv, session)
            if len(temp_exclusion_df) > 0:
                condition_acc_flags = pd.concat([condition_acc_flags, temp_exclusion_df], ignore_index=True)
        
        # Flag go_nogo accuracy
        for col_name_go in go_nogo_go_acc_cols:
            value = row[col_name_go]
            if pd.notna(value) and compare_to_threshold('go_acc', value, GO_ACC_THRESHOLD_GO_NOGO):
                condition_acc_flags = append_exclusion_row(condition_acc_flags, subject_id, col_name_go, value, GO_ACC_THRESHOLD_GO_NOGO, session)
        
        for col_name_nogo in go_nogo_nogo_acc_cols:
            value = row[col_name_nogo]
            if pd.notna(value) and compare_to_threshold('nogo_acc', value, NOGO_ACC_THRESHOLD_GO_NOGO):
                condition_acc_flags = append_exclusion_row(condition_acc_flags, subject_id, col_name_nogo, value, NOGO_ACC_THRESHOLD_GO_NOGO, session)
        
        # Flag other condition accuracies
        for col_name in acc_cols:
            value = row[col_name]
            if pd.notna(value) and compare_to_threshold(col_name, value, ACC_THRESHOLD):
                condition_acc_flags = append_exclusion_row(condition_acc_flags, subject_id, col_name, value, ACC_THRESHOLD, session)
        
        # Flag omission rates
        for col_name in omission_rate_cols:
            value = row[col_name]
            if pd.notna(value):
                # Use appropriate threshold based on column name
                if 'go_omission_rate' in col_name:
                    threshold = GO_OMISSION_RATE_THRESHOLD
                else:
                    threshold = OMISSION_RATE_THRESHOLD
                if compare_to_threshold(col_name, value, threshold):
                    omission_rate_flags = append_exclusion_row(omission_rate_flags, subject_id, col_name, value, threshold, session)
    
    return condition_acc_flags, omission_rate_flags
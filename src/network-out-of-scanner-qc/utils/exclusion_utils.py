import pandas as pd
import os
from pathlib import Path
import re
import numpy as np

from utils.globals import (
    STOP_SUCCESS_ACC_LOW_THRESHOLD,
    STOP_SUCCESS_ACC_HIGH_THRESHOLD,
    GO_RT_THRESHOLD,
    GO_RT_THRESHOLD_FMRI,
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

def append_exclusion_row(exclusion_df, subject_id, metric_name, metric_value, threshold, session=None):
    """Append a new exclusion row to the exclusion dataframe, avoiding duplicates."""
    # Check if this subject+metric combination already exists
    if len(exclusion_df) > 0:
        existing = exclusion_df[
            (exclusion_df['subject_id'] == subject_id) & 
            (exclusion_df['metric'] == metric_name)
        ]
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
    
    for index, row in task_csv.iterrows():
        #ignore the last 4 rows (summary rows)
        if index >= len(task_csv) - SUMMARY_ROWS:
            continue
        subject_id = row['subject_id']
        session = row['session'] if 'session' in row.index else None

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
                    rt_threshold = GO_RT_THRESHOLD_FMRI
                else:
                    rt_threshold = GO_RT_THRESHOLD if not is_dual_task(task_name) else GO_RT_THRESHOLD_DUAL_TASK
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

        # Check go_omission_rate columns - skip exclusion for fMRI (will be flagged instead)
        if not is_fmri:
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
    
    for index, row in task_csv.iterrows():
        #ignore the last 4 rows (summary rows)
        if index >= len(task_csv) - SUMMARY_ROWS:
            continue
        subject_id = row['subject_id']
        session = row['session'] if 'session' in row.index else None
        for load in [1, 2, 3]:
            cols = nback_get_columns(task_csv, load)
            exclusion_df = nback_flag_independent_accuracy(exclusion_df, subject_id, row, load, cols, session)
            # Check omission rates - skip exclusion for fMRI (will be flagged instead)
            if not is_fmri:
                exclusion_df = nback_flag_omission_rates(exclusion_df, subject_id, row, load, cols, session)
        # Check combined accuracy thresholds (once per row, not per load)
        exclusion_df = nback_flag_combined_accuracy(exclusion_df, subject_id, row, task_csv, session)

    #sort by subject_id
    if len(exclusion_df) != 0:
        exclusion_df = sort_subject_ids(exclusion_df)
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
        if subject_id == 's1351':
            print(f"mismatch_map: {mismatch_map}")
            print(f"match_map: {match_map}")
            print(f"common_suffixes: {common_suffixes}")

        for cond_suffix in common_suffixes:
            mismatch_col = mismatch_map[cond_suffix]
            match_col = match_map[cond_suffix]
            mismatch_val = row[mismatch_col]
            match_val = row[match_col]
            if pd.notna(mismatch_val) and pd.notna(match_val):
                if subject_id == 's1351' and match_val == 0.4:
                    print(f"went through pd.notna(mismatch_val) and pd.notna(match_val)")
                    print(f"mismatch_val: {mismatch_val}, match_val: {match_val}")
                    print(f"MISMATCH_COMBINED_THRESHOLD: {MISMATCH_COMBINED_THRESHOLD}, MATCH_COMBINED_THRESHOLD: {MATCH_COMBINED_THRESHOLD}")
                if (mismatch_val < MISMATCH_COMBINED_THRESHOLD) and (match_val < MATCH_COMBINED_THRESHOLD):
                    if subject_id == 's1351' and match_val == 0.4:
                        print(f"went through (mismatch_val < MISMATCH_COMBINED_THRESHOLD) and (match_val < MATCH_COMBINED_THRESHOLD)")
                    exclusion_df = append_exclusion_row(
                        exclusion_df, subject_id, f'{mismatch_col}_combined', mismatch_val, MISMATCH_COMBINED_THRESHOLD, session
                    )
                    print(f"exclusion_df: {exclusion_df}")
                    exclusion_df = append_exclusion_row(
                        exclusion_df, subject_id, f'{match_col}_combined', match_val, MATCH_COMBINED_THRESHOLD, session
                    )
                    print(f"exclusion_df: {exclusion_df}")
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
        exclusion_df = exclusion_df[~exclusion_df['metric'].str.contains('stop_success', na=False) & ~exclusion_df['metric'].str.contains('collapsed', na=False)]
    
    return exclusion_df

def create_combined_exclusions_csv(tasks, exclusions_output_path):
    """
    Create a combined exclusions CSV from all individual task exclusion files.
    
    Args:
        tasks (list): List of task names
        exclusions_output_path (Path): Path to the exclusions output folder
        
    Returns:
        None: Saves combined exclusions CSV to exclusions_output_path / 'all_exclusions.csv'
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
        combined_exclusions.to_csv(exclusions_output_path / 'all_exclusions.csv', index=False)
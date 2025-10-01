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
    MISMATCH_1_BACK_THRESHOLD,
    MISMATCH_2_BACK_THRESHOLD,
    MATCH_1_BACK_THRESHOLD,
    MATCH_2_BACK_THRESHOLD,
    MISMATCH_3_BACK_THRESHOLD,
    MATCH_3_BACK_THRESHOLD,
    ACC_THRESHOLD,
    OMISSION_RATE_THRESHOLD,
    LOWER_WEIGHT,
    UPPER_WEIGHT
)
from utils.qc_utils import sort_subject_ids

def check_exclusion_criteria(task_name, task_csv, exclusion_df):
        if 'stop_signal' in task_name:
            exclusion_df = check_stop_signal_exclusion_criteria(task_name, task_csv, exclusion_df)
        if 'go_nogo' in task_name:
            exclusion_df = check_go_nogo_exclusion_criteria(task_name, task_csv, exclusion_df)
        if 'n_back' in task_name:
            exclusion_df = check_n_back_exclusion_criteria(task_name, task_csv, exclusion_df)
        return exclusion_df
        # if ('flanker' in task_name or 'directed_forgetting' in task_name or 
        # 'shape_matching' in task_name or 'spatial_task_switching' in task_name or 
        # 'cued_task_switching' in task_name):
        #     check_other_exclusion_criteria(task_name, task_csv, exclusion_df)

def compare_to_threshold(metric_name, metric_value, threshold):
    """Check if a metric value violates the exclusion criteria."""
    return metric_value < threshold if 'low' in metric_name or 'acc' in metric_name else metric_value > threshold

def append_exclusion_row(exclusion_df, subject_id, metric_name, metric_value, threshold):
    """Append a new exclusion row to the exclusion dataframe."""
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
        if index >= len(task_csv) - 4:
            continue
        subject_id = row['subject_id']

        # Get actual column names for each metric type
        stop_success_cols = [col for col in task_csv.columns if 'stop_success' in col]
        go_rt_cols = [col for col in task_csv.columns if 'go_rt' in col]
        go_acc_cols = [col for col in task_csv.columns if 'go_acc' in col]
        go_omission_rate_cols = [col for col in task_csv.columns if 'go_omission_rate' in col]
        
        # Check stop_success specifically for low and high thresholds
        for col_name in stop_success_cols:
            value = row[col_name]
            if compare_to_threshold('stop_success_low', value, STOP_SUCCESS_ACC_LOW_THRESHOLD):
                exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, STOP_SUCCESS_ACC_LOW_THRESHOLD)
            if compare_to_threshold('stop_success_high', value, STOP_SUCCESS_ACC_HIGH_THRESHOLD):
                exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, STOP_SUCCESS_ACC_HIGH_THRESHOLD)

        # Check go_rt columns
        for col_name in go_rt_cols:
            value = row[col_name]
            if compare_to_threshold('go_rt', value, GO_RT_THRESHOLD):
                exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, GO_RT_THRESHOLD)

        # Check go_acc columns
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
        if index >= len(task_csv) - 4:
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
                    if compare_to_threshold('go_acc', go_acc_value, GO_ACC_THRESHOLD_GO_NOGO) and compare_to_threshold('_nogo_acc', nogo_acc_value, NOGO_ACC_THRESHOLD_GO_NOGO):
                        exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name_go, go_acc_value, GO_ACC_THRESHOLD_GO_NOGO)
                        exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name_nogo, nogo_acc_value, NOGO_ACC_THRESHOLD_GO_NOGO)
                    if np.mean([go_acc_value, nogo_acc_value]) < ACC_THRESHOLD:
                        exclusion_df = append_exclusion_row(exclusion_df, subject_id, 'mean_go_and_nogo_acc', np.mean([go_acc_value, nogo_acc_value]), ACC_THRESHOLD)

        for col_name in go_omission_rate_cols:
            value = row[col_name]
            if compare_to_threshold('go_omission_rate', value, OMISSION_RATE_THRESHOLD):
                exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, OMISSION_RATE_THRESHOLD)
    #sort by subject_id
    if len(exclusion_df) != 0:
        exclusion_df = sort_subject_ids(exclusion_df)
    return exclusion_df

def check_n_back_exclusion_criteria(task_name, task_csv, exclusion_df):
    for index, row in task_csv.iterrows():
        #ignore the last 4 rows (summary rows)
        if index >= len(task_csv) - 4:
            continue
        subject_id = row['subject_id']

        for level in [1, 2, 3]:
            cols = _nback_get_columns(task_csv, level)
            exclusion_df = _nback_flag_weighted_means(exclusion_df, subject_id, row, level, cols)
            exclusion_df = _nback_flag_joint_accuracy(exclusion_df, subject_id, row, level, cols)
            exclusion_df = _nback_flag_omission_rates(exclusion_df, subject_id, row, level, cols)

    #sort by subject_id
    if len(exclusion_df) != 0:
        exclusion_df = sort_subject_ids(exclusion_df)
    return exclusion_df

def _nback_get_columns(task_csv, level):
    level_str = f"{level}.0back"
    match_acc = [col for col in task_csv.columns if f'match_{level_str}' in col and 'acc' in col and 'mismatch' not in col]
    mismatch_acc = [col for col in task_csv.columns if f'mismatch_{level_str}' in col and 'acc' in col]
    match_omiss = [col for col in task_csv.columns if f'match_{level_str}' in col and 'omission_rate' in col and 'mismatch' not in col]
    mismatch_omiss = [col for col in task_csv.columns if f'mismatch_{level_str}' in col and 'omission_rate' in col]
    return {
        'match_acc': match_acc,
        'mismatch_acc': mismatch_acc,
        'match_omiss': match_omiss,
        'mismatch_omiss': mismatch_omiss,
    }

def _nback_iter_pairs(match_cols, mismatch_cols, level):
    match_prefix = f'match_{level}.0back'
    mismatch_prefix = f'mismatch_{level}.0back'
    match_map = {c.replace(match_prefix, ''): c for c in match_cols}
    mismatch_map = {c.replace(mismatch_prefix, ''): c for c in mismatch_cols}
    for suffix in set(match_map.keys()).intersection(mismatch_map.keys()):
        yield match_map[suffix], mismatch_map[suffix]

def _nback_flag_weighted_means(exclusion_df, subject_id, row, level, cols):
    for match_col, mismatch_col in _nback_iter_pairs(cols['match_acc'], cols['mismatch_acc'], level):
        match_val = row[match_col]
        mismatch_val = row[mismatch_col]
        if not pd.isna(match_val) and not pd.isna(mismatch_val):
            weighted = LOWER_WEIGHT * match_val + UPPER_WEIGHT * mismatch_val
            if compare_to_threshold(f'weighted_mean_{level}.0back_acc', weighted, ACC_THRESHOLD):
                exclusion_df = append_exclusion_row(
                    exclusion_df, subject_id, f'weighted_mean_{level}.0back_acc', weighted, ACC_THRESHOLD
                )
    return exclusion_df

def _nback_flag_joint_accuracy(exclusion_df, subject_id, row, level, cols):
    mismatch_threshold = {1: MISMATCH_1_BACK_THRESHOLD, 2: MISMATCH_2_BACK_THRESHOLD, 3: MISMATCH_3_BACK_THRESHOLD}[level]
    match_threshold = {1: MATCH_1_BACK_THRESHOLD, 2: MATCH_2_BACK_THRESHOLD, 3: MATCH_3_BACK_THRESHOLD}[level]
    for match_col, mismatch_col in _nback_iter_pairs(cols['match_acc'], cols['mismatch_acc'], level):
        mismatch_val = row[mismatch_col]
        match_val = row[match_col]
        if not pd.isna(mismatch_val) and not pd.isna(match_val):
            if compare_to_threshold(f'mismatch_{level}.0back_acc', mismatch_val, mismatch_threshold) and \
               compare_to_threshold(f'match_{level}.0back_acc', match_val, match_threshold):
                exclusion_df = append_exclusion_row(
                    exclusion_df, subject_id, f'mismatch_{level}.0back_acc', mismatch_val, mismatch_threshold
                )
                exclusion_df = append_exclusion_row(
                    exclusion_df, subject_id, f'match_{level}.0back_acc', match_val, match_threshold
                )
    return exclusion_df

def _nback_flag_omission_rates(exclusion_df, subject_id, row, level, cols):
    for match_om_col in cols['match_omiss']:
        val = row[match_om_col]
        if compare_to_threshold(f'match_{level}.0back_omission_rate', val, OMISSION_RATE_THRESHOLD):
            exclusion_df = append_exclusion_row(
                exclusion_df, subject_id, f'match_{level}.0back_omission_rate', val, OMISSION_RATE_THRESHOLD
            )
    for mismatch_om_col in cols['mismatch_omiss']:
        val = row[mismatch_om_col]
        if compare_to_threshold(f'mismatch_{level}.0back_omission_rate', val, OMISSION_RATE_THRESHOLD):
            exclusion_df = append_exclusion_row(
                exclusion_df, subject_id, f'mismatch_{level}.0back_omission_rate', val, OMISSION_RATE_THRESHOLD
            )
    return exclusion_df

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
    ACC_THRESHOLD,
    OMISSION_RATE_THRESHOLD
)
from utils.qc_utils import sort_subject_ids

def check_exclusion_criteria(task_name, task_csv, exclusion_df):
        if 'stop_signal' in task_name:
            exclusion_df = check_stop_signal_exclusion_criteria(task_name, task_csv, exclusion_df)
        if 'go_nogo' in task_name:
            exclusion_df = check_go_nogo_exclusion_criteria(task_name, task_csv, exclusion_df)
        return exclusion_df
        # if 'n_back' in task_name:
        #     check_n_back_exclusion_criteria(task_name, task_csv, exclusion_df)
        # if ('flanker' in task_name or 'directed_forgetting' in task_name or 
        # 'shape_matching' in task_name or 'spatial_task_switching' in task_name or 
        # 'cued_task_switching' in task_name):
        #     check_other_exclusion_criteria(task_name, task_csv, exclusion_df)

def compare_to_threshold(metric_name, metric_value, threshold):
    """Check if a metric value violates the exclusion criteria."""
    return metric_value < threshold if 'low' in metric_name or 'acc' in metric_name else metric_value > threshold

def append_exclusion_row(exclusion_df, subject_id, task_name, metric_name, metric_value, threshold):
    """Append a new exclusion row to the exclusion dataframe."""
    exclusion_df = pd.concat([
        exclusion_df,
        pd.DataFrame({
            'subject_id': [subject_id],
            'task_name': [task_name],
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
                exclusion_df = append_exclusion_row(exclusion_df, subject_id, task_name, col_name, value, STOP_SUCCESS_ACC_LOW_THRESHOLD)
            if compare_to_threshold('stop_success_high', value, STOP_SUCCESS_ACC_HIGH_THRESHOLD):
                exclusion_df = append_exclusion_row(exclusion_df, subject_id, task_name, col_name, value, STOP_SUCCESS_ACC_HIGH_THRESHOLD)

        # Check go_rt columns
        for col_name in go_rt_cols:
            value = row[col_name]
            if compare_to_threshold('go_rt', value, GO_RT_THRESHOLD):
                exclusion_df = append_exclusion_row(exclusion_df, subject_id, task_name, col_name, value, GO_RT_THRESHOLD)

        # Check go_acc columns
        for col_name in go_acc_cols:
            value = row[col_name]
            if compare_to_threshold('go_acc', value, ACC_THRESHOLD):
                exclusion_df = append_exclusion_row(exclusion_df, subject_id, task_name, col_name, value, ACC_THRESHOLD)

        # Check go_omission_rate columns
        for col_name in go_omission_rate_cols:
            value = row[col_name]
            if compare_to_threshold('go_omission_rate', value, OMISSION_RATE_THRESHOLD):
                exclusion_df = append_exclusion_row(exclusion_df, subject_id, task_name, col_name, value, OMISSION_RATE_THRESHOLD)

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
        go_acc_cols = [col for col in task_csv.columns if '_go_acc' in col]
        print(f"go_acc_cols: {go_acc_cols}")
        nogo_acc_cols = [col for col in task_csv.columns if '_nogo_acc' in col]
        print(f"nogo_acc_cols: {nogo_acc_cols}")
        go_omission_rate_cols = [col for col in task_csv.columns if '_go_omission_rate' in col]
        print(f"go_omission_rate_cols: {go_omission_rate_cols}")

        # If go accuracy < threshold AND nogo accuracy < threshold, then exclude
        # Only check when the prefix (before go_acc/nogo_acc) matches
        for col_name_go in go_acc_cols:
            for col_name_nogo in nogo_acc_cols:
                # Extract prefix before 'go_acc' and 'nogo_acc'
                go_prefix = col_name_go.replace('_go_acc', '')
                nogo_prefix = col_name_nogo.replace('_nogo_acc', '')
                print(f"go_prefix: {go_prefix}, nogo_prefix: {nogo_prefix}")
                
                # Only proceed if prefixes match
                if go_prefix == nogo_prefix:
                    go_acc_value = row[col_name_go]
                    nogo_acc_value = row[col_name_nogo]
                    print(f"go_acc_value: {go_acc_value}, nogo_acc_value: {nogo_acc_value}")
                    if compare_to_threshold('go_acc', go_acc_value, GO_ACC_THRESHOLD_GO_NOGO) and compare_to_threshold('_nogo_acc', nogo_acc_value, NOGO_ACC_THRESHOLD_GO_NOGO):
                        exclusion_df = append_exclusion_row(exclusion_df, subject_id, task_name, col_name_go, go_acc_value, GO_ACC_THRESHOLD_GO_NOGO)
                        exclusion_df = append_exclusion_row(exclusion_df, subject_id, task_name, col_name_nogo, nogo_acc_value, NOGO_ACC_THRESHOLD_GO_NOGO)
                    if np.mean([go_acc_value, nogo_acc_value]) < ACC_THRESHOLD:
                        exclusion_df = append_exclusion_row(exclusion_df, subject_id, task_name, [col_name_go, col_name_nogo], np.mean([go_acc_value, nogo_acc_value]), ACC_THRESHOLD)

        for col_name in go_omission_rate_cols:
            value = row[col_name]
            if compare_to_threshold('go_omission_rate', value, OMISSION_RATE_THRESHOLD):
                exclusion_df = append_exclusion_row(exclusion_df, subject_id, task_name, col_name, value, OMISSION_RATE_THRESHOLD)
    #sort by subject_id
    print(exclusion_df.head())
    exclusion_df = sort_subject_ids(exclusion_df)
    return exclusion_df

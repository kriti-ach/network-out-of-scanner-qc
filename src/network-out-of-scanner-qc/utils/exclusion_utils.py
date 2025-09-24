import pandas as pd
import os
from pathlib import Path
import re
import numpy as np

from utils.globals import (
    STOP_ACC_LOW_THRESHOLD,
    STOP_ACC_HIGH_THRESHOLD,
    GO_RT_THRESHOLD,
    GO_ACC_THRESHOLD,
    NOGO_ACC_THRESHOLD,
    MISMATCH_1_BACK_THRESHOLD,
    MISMATCH_2_BACK_THRESHOLD,
    MATCH_1_BACK_THRESHOLD,
    MATCH_2_BACK_THRESHOLD,
    ACC_THRESHOLD,
    OMISSION_RATE_THRESHOLD
)

def check_exclusion_criteria(task_name, task_csv, exclusion_df):
        if 'stop_signal' in task_name:
            exclusion_df = check_stop_signal_exclusion_criteria(task_name, task_csv, exclusion_df)
        return exclusion_df
        # if 'go_nogo' in task_name:
        #     check_go_nogo_exclusion_criteria(task_name, task_csv, exclusion_df)
        # if 'n_back' in task_name:
        #     check_n_back_exclusion_criteria(task_name, task_csv, exclusion_df)
        # if ('flanker' in task_name or 'directed_forgetting' in task_name or 
        # 'shape_matching' in task_name or 'spatial_task_switching' in task_name or 
        # 'cued_task_switching' in task_name):
        #     check_other_exclusion_criteria(task_name, task_csv, exclusion_df)

def compare_to_threshold(metric_name, metric_value, threshold):
    """Check if a metric value violates the exclusion criteria."""
    return metric_value < threshold if 'low' in metric_name else metric_value > threshold

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
        subject_id = row['subject_id']

        # Create a dictionary to hold metric values from the row
        metrics_info = {
            'stop_fail_acc': [row.filter(like='stop_fail_acc').values, 
                              STOP_ACC_LOW_THRESHOLD, 
                              STOP_ACC_HIGH_THRESHOLD],
            'go_rt': [row.filter(like='go_rt').values[0], GO_RT_THRESHOLD],
            'go_acc': [row.filter(like='go_acc').values[0], GO_ACC_THRESHOLD],
            'go_omission_rate': [row.filter(like='go_omission_rate').values[0], OMISSION_RATE_THRESHOLD]
        }
        
        # Check stop_fail_acc specifically for low and high thresholds
        stop_fail_acc_values = metrics_info['stop_fail_acc'][0]
        for value in stop_fail_acc_values:
            if compare_to_threshold('stop_fail_acc_low', value, STOP_ACC_LOW_THRESHOLD):
                exclusion_df = append_exclusion_row(exclusion_df, subject_id, task_name, 'stop_fail_acc', value, STOP_ACC_LOW_THRESHOLD)
            if compare_to_threshold('stop_fail_acc_high', value, STOP_ACC_HIGH_THRESHOLD):
                exclusion_df = append_exclusion_row(exclusion_df, subject_id, task_name, 'stop_fail_acc', value, STOP_ACC_HIGH_THRESHOLD)

        # Check other metrics
        for metric_name, (metric_value, *thresholds) in metrics_info.items():
            # If multiple thresholds are provided, check each one
            for threshold in thresholds:
                if metric_value.size > 0:  # Ensure there are values to check
                    for value in metric_value:
                        if compare_to_threshold(metric_name, value, threshold):
                            exclusion_df = append_exclusion_row(exclusion_df, subject_id, task_name, metric_name, value, threshold)

    return exclusion_df






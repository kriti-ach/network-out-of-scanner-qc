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
            'stop_success': [row.filter(like='stop_success').values,
                              STOP_SUCCESS_ACC_LOW_THRESHOLD,
                              STOP_SUCCESS_ACC_HIGH_THRESHOLD],
            'go_rt': [np.array(row.filter(like='go_rt').values).flatten(), GO_RT_THRESHOLD],
            'go_acc': [np.array(row.filter(like='go_acc').values).flatten(), ACC_THRESHOLD],
            'go_omission_rate': [np.array(row.filter(like='go_omission_rate').values).flatten(), OMISSION_RATE_THRESHOLD]
        }
        
        # Check stop_success specifically for low and high thresholds
        stop_success_values = metrics_info['stop_success'][0]
        for value in stop_success_values:
            if compare_to_threshold('stop_success_low', value, STOP_SUCCESS_ACC_LOW_THRESHOLD):
                exclusion_df = append_exclusion_row(exclusion_df, subject_id, task_name, 'stop_success', value, STOP_SUCCESS_ACC_LOW_THRESHOLD)
            if compare_to_threshold('stop_success_high', value, STOP_SUCCESS_ACC_HIGH_THRESHOLD):
                exclusion_df = append_exclusion_row(exclusion_df, subject_id, task_name, 'stop_success', value, STOP_SUCCESS_ACC_HIGH_THRESHOLD)

        # Check other metrics
        for metric_name, (metric_value, *thresholds) in metrics_info.items():
            # If metric_value is already a numpy array, we don't need to check its size
            for threshold in thresholds:
                # Iterate over the metric values (this handles both arrays and single values)
                for value in np.atleast_1d(metric_value):  # Treat metric_value as an array
                    if compare_to_threshold(metric_name, value, threshold):
                        exclusion_df = append_exclusion_row(exclusion_df, subject_id, task_name, metric_name, value, threshold)

    return exclusion_df






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
            check_stop_signal_exclusion_criteria(task_name, task_csv, exclusion_df)
        if 'go_nogo' in task_name:
            check_go_nogo_exclusion_criteria(task_name, task_csv, exclusion_df)
        if 'n_back' in task_name:
            check_n_back_exclusion_criteria(task_name, task_csv, exclusion_df)
        if ('flanker' in task_name or 'directed_forgetting' in task_name or 
        'shape_matching' in task_name or 'spatial_task_switching' in task_name or 
        'cued_task_switching' in task_name):
            check_other_exclusion_criteria(task_name, task_csv, exclusion_df)

def check_stop_signal_exclusion_criteria(task_name, task_csv, exclusion_df):
    for index, row in task_csv.iterrows():
        subject_id = row['subject_id']
        if row['stop_fail_acc'] < STOP_ACC_LOW_THRESHOLD:
            exclusion_df = pd.concat([exclusion_df, pd.DataFrame({'subject_id': [subject_id], 'task_name': [task_name], 
            'metric': ['stop_fail_acc'], 'metric_value': [row['stop_fail_acc']], 
            'threshold': [STOP_ACC_LOW_THRESHOLD]})], ignore_index=True)
        if row['stop_fail_acc'] > STOP_ACC_HIGH_THRESHOLD:


def check_go_nogo_exclusion_criteria(task_csv):




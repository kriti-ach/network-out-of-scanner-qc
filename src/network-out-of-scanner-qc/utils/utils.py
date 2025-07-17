import pandas as pd
import os
from pathlib import Path
import re
import numpy as np

from utils.globals import (
    DUAL_TASKS_OUT_OF_SCANNER,
    SINGLE_TASKS_OUT_OF_SCANNER,
    DUAL_TASKS_FMRI,
    SINGLE_TASKS_FMRI,
    FLANKER_CONDITIONS,
    DIRECTED_FORGETTING_CONDITIONS,
    SPATIAL_TASK_SWITCHING_CONDITIONS,
    CUED_TASK_SWITCHING_CONDITIONS,
    SPATIAL_WITH_CUED_CONDITIONS,
    STOP_SIGNAL_CONDITIONS
)

def initialize_qc_csvs(tasks, output_path):
    """
    Initialize QC CSV files for all tasks.
    
    Args:
        tasks (list): List of task names
        output_path (Path): Path to save QC CSVs
    """
    for task in tasks:
        columns = get_task_columns(task)
        df = pd.DataFrame(columns=columns)
        df.to_csv(output_path / f"{task}_qc.csv", index=False)

def extend_metric_columns(base_columns, conditions):
    """
    Extend base columns with accuracy and RT metrics for given conditions.
    
    Args:
        base_columns (list): Base columns (e.g., ['subject_id', 'session', 'run'])
        conditions (list): List of conditions to create metrics for
        
    Returns:
        list: Extended list of columns with _acc and _rt for each condition
    """
    metric_types = ['acc', 'rt', 'omission_rate', 'commission_rate']
    return base_columns + [
        f'{cond}_{metric}' 
        for cond in conditions 
        for metric in metric_types
    ]

def get_task_columns(task_name, sample_df=None):
    """
    Define columns for each task's QC CSV.
    """
    base_columns = ['subject_id', 'session', 'run']
    
    if is_dual_task(task_name):
        if 'directed_forgetting' in task_name and 'flanker' in task_name or 'directedForgetting' in task_name and 'flanker' in task_name:
            # For dual tasks, create combined condition names
            conditions = [
                f'{df_cond}_{f_cond}'
                for df_cond in DIRECTED_FORGETTING_CONDITIONS
                for f_cond in FLANKER_CONDITIONS
            ]
            return extend_metric_columns(base_columns, conditions)
            
        elif 'cued_task_switching' in task_name and 'spatial_task_switching' in task_name or 'CuedTS' in task_name and 'spatialTS' in task_name:
            return extend_metric_columns(base_columns, SPATIAL_WITH_CUED_CONDITIONS)
    else:
        if 'spatial_task_switching' in task_name or 'spatialTS' in task_name:
            return extend_metric_columns(base_columns, SPATIAL_TASK_SWITCHING_CONDITIONS)
        elif 'cued_task_switching' in task_name or 'cuedTS' in task_name:
            return extend_metric_columns(base_columns, CUED_TASK_SWITCHING_CONDITIONS)
        else:
            print(f"Unknown task: {task_name}")
            return None

def is_dual_task(task_name):
    """
    Check if the task is a dual task.
    """
    # return any(task in task_name for task in DUAL_TASKS_OUT_OF_SCANNER)
    return any(task in task_name for task in DUAL_TASKS_FMRI)

def extract_task_name_out_of_scanner(filename):
    """
    Extract task name from filename using regex pattern.
    
    Args:
        filename (str): Name of the file
        
    Returns:
        str: Extracted task name or None if pattern doesn't match
    """
    match = re.match(r"s\d{2,}_(.*)\.csv", filename)
    if match:
        return match.group(1)
    return None

def extract_task_name_fmri(filename):
    """
    Extract task name from filename using regex pattern.
    
    Args:
        filename (str): Name of the file
        
    Returns:
        str: Extracted task name or None if pattern doesn't match
    """
    match = re.match(r"sub-s\d{2,}_ses-(.*)_task-(.*)_run-(.*)_events\.tsv", filename)
    if match:
        return match.group(2)
    return None

def filter_to_test_trials(df, task_name):
    """
    Filter the dataframe to only include test trials.
    
    Args:
        df (pd.DataFrame): Input dataframe
        task_name (str): Name of the task
        
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    return df[df['trial_id'] == 'test_trial']

def update_qc_csv(output_path, task_name, subject_id, metrics):
    """
    Update the QC CSV file for a specific task with new data.
    
    Args:
        output_path (Path): Path to save QC CSVs
        task_name (str): Name of the task
        subject_id (str): Subject ID
        metrics (dict): Dictionary of metrics to add
    """
    qc_file = output_path / f"{task_name}_qc.csv"
    try:
        df = pd.read_csv(qc_file)
        new_row = pd.DataFrame({
            'subject_id': [subject_id],
            **metrics
        })
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(qc_file, index=False)
    except FileNotFoundError:
        print(f"Warning: QC file {qc_file} not found") 

def get_task_metrics(df, task_name):
    """
    Main function to get metrics for any task.
    
    Args:
        df (pd.DataFrame): DataFrame containing task data
        task_name (str): Name of the task
        
    Returns:
        dict: Dictionary containing task-specific metrics
    """
    # First filter to test trials
    df = filter_to_test_trials(df, task_name)
    
    if is_dual_task(task_name):
        # For dual tasks, we need both sets of conditions
        if ('directed_forgetting' in task_name and 'flanker' in task_name) or ('directedForgetting' in task_name and 'flanker' in task_name):
            conditions = {
                'directed_forgetting': DIRECTED_FORGETTING_CONDITIONS,
                'flanker': FLANKER_CONDITIONS
            }
            condition_columns = {
                'directed_forgetting': 'directed_forgetting_condition',
                'flanker': 'flanker_condition'
            }
            return calculate_metrics(df, conditions, condition_columns, is_dual_task(task_name))
        
        elif ('cued_task_switching' in task_name and 'spatial_task_switching' in task_name) or ('CuedTS' in task_name and 'spatialTS' in task_name):
            metrics = {}
            for cond in SPATIAL_WITH_CUED_CONDITIONS:
                mask_acc = (df['task_switch'] == cond)
                mask_rt = (df['task_switch'] == cond) & (df['correct_trial'] == 1)
                mask_omission = (df['task_switch'] == cond) & (df['key_press'] == -1)
                mask_commission = (df['task_switch'] == cond) & (df['key_press'] != -1) & (df['correct_trial'] == 0)
                num_omissions = len(df[mask_omission])
                num_commissions = len(df[mask_commission])
                total_num_trials = len(df[mask_acc])
                metrics[f'{cond}_acc'] = df[mask_acc]['correct_trial'].mean()
                metrics[f'{cond}_rt'] = df[mask_rt]['rt'].mean()
                metrics[f'{cond}_omission_rate'] = num_omissions / total_num_trials
                metrics[f'{cond}_commission_rate'] = num_commissions / total_num_trials
            return metrics
    else:
        # Special handling for n-back task
        if 'n_back' in task_name:
            metrics = {}
            # Get unique combinations of trial_type and delay
            for trial_type in df['trial_type'].unique():
                for delay in df['delay'].unique():
                    condition = f"{trial_type}_{delay}back"
                    mask_acc = (df['trial_type'] == trial_type) & (df['delay'] == delay)
                    mask_rt = mask_acc & (df['correct_trial'] == 1)
                    metrics[f'{condition}_acc'] = df[mask_acc]['correct_trial'].mean()
                    metrics[f'{condition}_rt'] = df[mask_rt]['rt'].mean()
            return metrics
            
        # Special handling for stop signal task
        elif 'stop_signal' in task_name:
            metrics = {}
            # Calculate metrics for each condition
            for condition in STOP_SIGNAL_CONDITIONS:
                mask_acc = (df['trial_type'] == condition)
                mask_rt = mask_acc & (df['correct_trial'] == 1)
                metrics[f'{condition}_acc'] = df[mask_acc]['correct_trial'].mean()
                metrics[f'{condition}_rt'] = df[mask_rt]['rt'].mean()
                # Add count for stop trials
                if condition in ['stop_success', 'stop_failure']:
                    metrics[f'{condition}_count'] = len(df[mask_acc])
            
            # Add SS_delay statistics
            metrics['mean_SSD'] = df[df['SS_delay'].notna()]['SS_delay'].mean()
            metrics['std_SSD'] = df[df['SS_delay'].notna()]['SS_delay'].std()
            return metrics
            
        # For other single tasks, we only need one set of conditions
        elif 'directed_forgetting' in task_name or 'directedForgetting' in task_name:
            conditions = {'directed_forgetting': DIRECTED_FORGETTING_CONDITIONS}
            condition_columns = {'directed_forgetting': 'directed_forgetting_condition'}
        elif 'flanker' in task_name:
            conditions = {'flanker': FLANKER_CONDITIONS}
            condition_columns = {'flanker': 'flanker_condition'}
        elif 'spatial_task_switching' in task_name or 'spatialTS' in task_name:
            conditions = {'spatial_task_switching': SPATIAL_TASK_SWITCHING_CONDITIONS}
            condition_columns = {'spatial_task_switching': 'trial_type'}
        elif 'cued_task_switching' in task_name or 'cuedTS' in task_name:
            conditions = {'cued_task_switching': CUED_TASK_SWITCHING_CONDITIONS}
            condition_columns = {'cued_task_switching': 'trial_type'}
        else:
            print(f"Unknown task: {task_name}")
            return None
    
        return calculate_metrics(df, conditions, condition_columns, is_dual_task(task_name))

def calculate_metrics(df, conditions, condition_columns, is_dual_task):
    """
    Calculate RT and accuracy metrics for any task.
    
    Args:
        df (pd.DataFrame): DataFrame containing task data
        conditions (dict): Dictionary of task names and their conditions
        condition_columns (dict): Dictionary of task names and their condition column names
        is_dual_task (bool): Whether this is a dual task
        
    Returns:
        dict: Dictionary containing task-specific metrics
    """
    metrics = {}
    
    if is_dual_task:
        # For dual tasks, iterate through all combinations of conditions
        task1, task2 = list(conditions.keys())
        for cond1 in conditions[task1]:
            for cond2 in conditions[task2]:
                mask_acc = (df[condition_columns[task1]] == cond1) & (df[condition_columns[task2]] == cond2)
                mask_rt = (df[condition_columns[task1]] == cond1) & (df[condition_columns[task2]] == cond2) & (df['correct_trial'] == 1)
                metrics[f'{cond1}_{cond2}_acc'] = df[mask_acc]['correct_trial'].mean()
                metrics[f'{cond1}_{cond2}_rt'] = df[mask_rt]['rt'].mean()
    else:
        # For single tasks, just iterate through conditions
        task = list(conditions.keys())[0]
        for cond in conditions[task]:
            mask_acc = (df[condition_columns[task]] == cond)
            mask_rt = (df[condition_columns[task]] == cond) & (df['correct_trial'] == 1)
            metrics[f'{cond}_acc'] = df[mask_acc]['correct_trial'].mean()
            metrics[f'{cond}_rt'] = df[mask_rt]['rt'].mean()
    
    return metrics

def append_summary_rows_to_csv(csv_path):
    # Skip if file is empty or has no columns
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return
    if df.empty or len(df.columns) < 4:
        return
    # Only operate if there are at least 4 columns
    stats_cols = df.columns[3:]
    summary = {
        'mean': ['mean', np.nan, np.nan] + [df[col].mean() for col in stats_cols],
        'std':  ['std', np.nan, np.nan] + [df[col].std() for col in stats_cols],
        'max':  ['max', np.nan, np.nan] + [df[col].max() for col in stats_cols],
        'min':  ['min', np.nan, np.nan] + [df[col].min() for col in stats_cols],
    }
    for stat, values in summary.items():
        row = pd.Series(values, index=df.columns, name=stat)
        df = pd.concat([df, row.to_frame().T], ignore_index=True)
    df.to_csv(csv_path, index=False)
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
    SPATIAL_WITH_CUED_CONDITIONS
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

def get_task_columns(task_name, sample_df=None):
    """
    Define columns for each task's QC CSV.
    If sample_df is provided, use it for dynamic contrast extraction (e.g., cued+spatialts).
    """
    if is_dual_task(task_name):
        if 'directed_forgetting' in task_name and 'flanker' in task_name or 'directedForgetting' in task_name and 'flanker' in task_name:
            columns = ['subject_id', 'session', 'run']
            for df_cond in DIRECTED_FORGETTING_CONDITIONS:
                for flanker_cond in FLANKER_CONDITIONS:
                    columns.extend([
                        f'{df_cond}_{flanker_cond}_acc',
                        f'{df_cond}_{flanker_cond}_rt'
                    ])
            return columns
        elif 'cued_task_switching' in task_name and 'spatial_task_switching' in task_name or 'CuedTS' in task_name and 'spatialTS' in task_name:
            columns = ['subject_id', 'session', 'run']
            for cond in SPATIAL_WITH_CUED_CONDITIONS:
                columns.extend([f'{cond}_acc', f'{cond}_rt'])
            return columns
    else:
        if 'spatial_task_switching' in task_name or 'spatialTS' in task_name:
            columns = ['subject_id', 'session', 'run']
            for cond in SPATIAL_TASK_SWITCHING_CONDITIONS:
                columns.extend([f'{cond}_acc', f'{cond}_rt'])
            return columns
        elif 'cued_task_switching' in task_name or 'cuedTS' in task_name:
            columns = ['subject_id', 'session', 'run']
        elif 'spatial_task_switching' in task_name or 'spatialTS' in task_name:
            columns = ['subject_id', 'session', 'run']
            for cond in SPATIAL_TASK_SWITCHING_CONDITIONS:
                columns.extend([f'{cond}_acc', f'{cond}_rt'])
            return columns
        elif 'cued_task_switching' in task_name or 'cuedTS' in task_name:
            columns = ['subject_id', 'session', 'run']
            for cond in CUED_TASK_SWITCHING_CONDITIONS:
                columns.extend([f'{cond}_acc', f'{cond}_rt'])
            return columns
        elif 'directed_forgetting' in task_name or 'directedForgetting' in task_name:
            columns = ['subject_id', 'session', 'run']
            for cond in DIRECTED_FORGETTING_CONDITIONS:
                columns.extend([f'{cond}_acc', f'{cond}_rt'])
            return columns
        elif 'flanker' in task_name:
            columns = ['subject_id', 'session', 'run']
            for cond in FLANKER_CONDITIONS:
                columns.extend([f'{cond}_acc', f'{cond}_rt'])
            return columns
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

def update_qc_csv(output_path, task_name, subject_id, session, run, metrics):
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
            'session': [session],
            'run': [run],
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
                mask_rt = (df['task_switch'] == cond) & (df['acc'] == 1)
                metrics[f'{cond}_acc'] = df[mask_acc]['acc'].mean()
                metrics[f'{cond}_rt'] = df[mask_rt]['response_time'].mean()
            
            return metrics
    else:
        # For single tasks, we only need one set of conditions
        if 'directed_forgetting' in task_name or 'directedForgetting' in task_name:
            conditions = {'directed_forgetting': DIRECTED_FORGETTING_CONDITIONS}
            condition_columns = {'directed_forgetting': 'directed_forgetting_condition'}
        elif 'flanker' in task_name or 'flanker' in task_name:
            conditions = {'flanker': FLANKER_CONDITIONS}
            condition_columns = {'flanker': 'flanker_condition'}
        elif 'spatial_task_switching' in task_name or 'spatialTS' in task_name:
            conditions = {'spatial_task_switching': SPATIAL_TASK_SWITCHING_CONDITIONS}
            condition_columns = {'spatial_task_switching': 'trial_type'}
        elif 'cued_task_switching' in task_name or 'cuedTS' in task_name:
            conditions = {'cued_task_switching': CUED_TASK_SWITCHING_CONDITIONS}
            condition_columns = {'cued_task_switching': 'trial_type'}
    
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
                mask = (
                    (df[condition_columns[task1]] == cond1) & 
                    (df[condition_columns[task2]] == cond2)
                )
                mask_acc = (df[condition_columns[task1]] == cond1) & (df[condition_columns[task2]] == cond2)
                mask_rt = (df[condition_columns[task1]] == cond1) & (df[condition_columns[task2]] == cond2) & (df['acc'] == 1)
                metrics[f'{cond1}_{cond2}_acc'] = df[mask_acc]['acc'].mean()
                metrics[f'{cond1}_{cond2}_rt'] = df[mask_rt]['response_time'].mean()
    else:
        # For single tasks, just iterate through conditions
        task = list(conditions.keys())[0]
        for cond in conditions[task]:
            mask = (df[condition_columns[task]] == cond)
            mask_acc = (df[condition_columns[task]] == cond)
            mask_rt = (df[condition_columns[task]] == cond) & (df['acc'] == 1)
            metrics[f'{cond}_acc'] = df[mask_acc]['acc'].mean()
            metrics[f'{cond}_rt'] = df[mask_rt]['response_time'].mean()
    
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
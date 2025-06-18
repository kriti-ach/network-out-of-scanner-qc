import pandas as pd
import os
from pathlib import Path
import re
import numpy as np

from utils.globals import (
    DUAL_TASKS,
    SINGLE_TASKS,
    FLANKER_CONDITIONS,
    DIRECTED_FORGETTING_CONDITIONS,
    SPATIAL_TASK_SWITCHING_CONDITIONS,
    CUED_TASK_SWITCHING_CONDITIONS
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

def get_cued_spatialts_contrasts(df):
    """
    Get unique, non-empty, non-na contrast names from the task_switch column.
    """
    return [v for v in df['task_switch'].unique() if pd.notna(v) and v != '']

def get_task_columns(task_name, sample_df=None):
    """
    Define columns for each task's QC CSV.
    If sample_df is provided, use it for dynamic contrast extraction (e.g., cued+spatialts).
    """
    if is_dual_task(task_name):
        if 'directed_forgetting' in task_name and 'flanker' in task_name:
            columns = ['subject_id']
            for df_cond in DIRECTED_FORGETTING_CONDITIONS:
                for flanker_cond in FLANKER_CONDITIONS:
                    columns.extend([
                        f'{df_cond}_{flanker_cond}_acc',
                        f'{df_cond}_{flanker_cond}_rt'
                    ])
            return columns
        elif 'cued_task_switching' in task_name and 'spatial_task_switching' in task_name:
            # Do not create columns at init; handled dynamically in main
            return None
    else:
        if 'spatial_task_switching' in task_name:
            columns = ['subject_id']
            for cond in SPATIAL_TASK_SWITCHING_CONDITIONS:
                columns.extend([f'{cond}_acc', f'{cond}_rt'])
            return columns
        elif 'cued_task_switching' in task_name:
            columns = ['subject_id']
        elif 'spatial_task_switching' in task_name:
            columns = ['subject_id']
            for cond in SPATIAL_TASK_SWITCHING_CONDITIONS:
                columns.extend([f'{cond}_acc', f'{cond}_rt'])
            return columns
        elif 'cued_task_switching' in task_name:
            columns = ['subject_id']
            for cond in CUED_TASK_SWITCHING_CONDITIONS:
                columns.extend([f'{cond}_acc', f'{cond}_rt'])
            return columns
        elif 'directed_forgetting' in task_name:
            columns = ['subject_id']
            for cond in DIRECTED_FORGETTING_CONDITIONS:
                columns.extend([f'{cond}_acc', f'{cond}_rt'])
            return columns
        elif 'flanker' in task_name:
            columns = ['subject_id']
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
    return any(task in task_name for task in DUAL_TASKS)

def extract_task_name(filename):
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
    if 'cued_task_switching' in task_name and 'spatial_task_switching' in task_name:
        create_cued_spatialts_csv(task_name, df, output_path)
        return
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
        if 'directed_forgetting' in task_name and 'flanker' in task_name:
            conditions = {
                'directed_forgetting': DIRECTED_FORGETTING_CONDITIONS,
                'flanker': FLANKER_CONDITIONS
            }
            condition_columns = {
                'directed_forgetting': 'directed_forgetting_condition',
                'flanker': 'flanker_condition'
            }
            return calculate_metrics(df, conditions, condition_columns, is_dual_task(task_name))
        
        elif 'cued_task_switching' in task_name and 'spatial_task_switching' in task_name:
            contrasts = get_cued_spatialts_contrasts(df)
            print(f'contrasts: {contrasts}')
            metrics = {}
            for contrast in contrasts:
                mask = (df['task_switch'] == contrast)
                metrics[f'{contrast}_acc'] = df[mask]['correct_trial'].mean()
                metrics[f'{contrast}_rt'] = df[mask]['rt'].mean()
            print(f'metrics: {metrics}')
            return metrics
    else:
        # For single tasks, we only need one set of conditions
        if 'directed_forgetting' in task_name:
            conditions = {'directed_forgetting': DIRECTED_FORGETTING_CONDITIONS}
            condition_columns = {'directed_forgetting': 'directed_forgetting_condition'}
        elif 'flanker' in task_name:
            conditions = {'flanker': FLANKER_CONDITIONS}
            condition_columns = {'flanker': 'flanker_condition'}
        elif 'spatial_task_switching' in task_name:
            conditions = {'spatial_task_switching': SPATIAL_TASK_SWITCHING_CONDITIONS}
            condition_columns = {'spatial_task_switching': 'spatial_task_switching_condition'}
        elif 'cued_task_switching' in task_name:
            conditions = {'cued_task_switching': CUED_TASK_SWITCHING_CONDITIONS}
            condition_columns = {'cued_task_switching': 'cued_task_switching_condition'}
    
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
                metrics[f'{cond1}_{cond2}_acc'] = df[mask]['correct_trial'].mean()
                metrics[f'{cond1}_{cond2}_rt'] = df[mask]['rt'].mean()
    else:
        # For single tasks, just iterate through conditions
        task = list(conditions.keys())[0]
        for cond in conditions[task]:
            mask = (df[condition_columns[task]] == cond)
            metrics[f'{cond}_acc'] = df[mask]['correct_trial'].mean()
            metrics[f'{cond}_rt'] = df[mask]['rt'].mean()
    
    return metrics

# Helper to create cued+spatialts CSV dynamically
def create_cued_spatialts_csv(task_name, df, output_path):
    contrasts = get_cued_spatialts_contrasts(df)
    columns = ['subject_id']
    for contrast in contrasts:
        columns.extend([f'{contrast}_acc', f'{contrast}_rt'])
    pd.DataFrame(columns=columns).to_csv(output_path / f"{task_name}_qc.csv", index=False)
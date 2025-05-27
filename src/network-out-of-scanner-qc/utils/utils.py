import pandas as pd
import os
from pathlib import Path
import re
import numpy as np

from utils.globals import (
    TASKS
)

def initialize_qc_csvs(tasks, output_path):
    """
    Initialize QC CSV files for all tasks.
    
    Args:
        tasks (list): List of task names
    """
    for task in tasks:
        # Get example metrics to determine columns
        example_metrics = get_task_metrics(pd.DataFrame(), task)
        columns = ['subject_id'] + list(example_metrics.keys())
        df = pd.DataFrame(columns=columns)
        df.to_csv(output_path / f"{task}_qc.csv", index=False)

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
    """
    return df[df['trial_id'] == 'test_trial']

def update_qc_csv(output_path, task_name, subject_id, metrics):
    """
    Update the QC CSV file for a specific task with new data.
    
    Args:
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

def calculate_df_with_flanker_metrics(df, task1, task2):
    """
    Calculate RT and accuracy metrics for dual tasks.
    
    Args:
        df (pd.DataFrame): DataFrame containing dual task data
        task1 (str): Name of first task (e.g., 'directed_forgetting')
        task2 (str): Name of second task (e.g., 'flanker')
        
    Returns:
        dict: Dictionary containing RT and accuracy metrics for each condition combination
    """
    metrics = {}
    
    # Example for directed forgetting + flanker
    if task1 == 'directed_forgetting' and task2 == 'flanker':
        for df_cond in ['con', 'pos', 'neg']:
            for flanker_cond in ['congruent', 'incongruent']:
                mask = (df['directed_forgetting_condition'] == df_cond) & (df['flanker_condition'] == flanker_cond)
                metrics[f'{df_cond}_{flanker_cond}_acc'] = df[mask]['correct'].mean()
                metrics[f'{df_cond}_{flanker_cond}_rt'] = df[mask]['rt'].mean()
    
    return metrics

def get_task_metrics(df, task_name):
    """
    Main function to get metrics for any task.
    
    Args:
        df (pd.DataFrame): DataFrame containing task data
        task_name (str): Name of the task
        
    Returns:
        dict: Dictionary containing task-specific metrics
    """
    if 'directed_forgetting' in task_name and 'flanker' in task_name:
        return calculate_df_with_flanker_metrics(df, 'directed_forgetting', 'flanker')
    else:
        raise ValueError(f"Unknown task: {task_name}") 
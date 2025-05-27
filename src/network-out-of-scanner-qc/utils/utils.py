import pandas as pd
import os
from pathlib import Path
import re
import numpy as np
from .task_metrics import get_task_metrics

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
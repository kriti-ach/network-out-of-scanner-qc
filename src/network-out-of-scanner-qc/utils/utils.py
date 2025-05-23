import pandas as pd
from pathlib import Path
import re

from utils.globals import (
    SINGLE_TASKS,
    DUAL_TASKS
)

def initialize_qc_csvs():
    """Initialize QC CSV files for all tasks."""
    for task in SINGLE_TASKS + DUAL_TASKS:
        df = pd.DataFrame(columns=["subject_id", "task", "score"])
        df.to_csv(f"{task}_qc.csv", index=False)

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

def update_qc_csv(task_name, subject_id, score):
    """
    Update the QC CSV file for a specific task with new data.
    
    Args:
        task_name (str): Name of the task
        subject_id (str): Subject ID
        score (float): Score to be added
    """
    qc_file = f"{task_name}_qc.csv"
    try:
        df = pd.read_csv(qc_file)
        new_row = pd.DataFrame({
            "subject_id": [subject_id],
            "task": [task_name],
            "score": [score]
        })
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(qc_file, index=False)
    except FileNotFoundError:
        print(f"Warning: QC file {qc_file} not found") 
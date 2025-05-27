import pandas as pd
import numpy as np
from .utils import filter_to_test_trials

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
    df = filter_to_test_trials(df, task_name)
    if 'directed_forgetting' in task_name and 'flanker' in task_name:
        return calculate_df_with_flanker_metrics(df, 'directed_forgetting', 'flanker')
    else:
        raise ValueError(f"Unknown task: {task_name}") 
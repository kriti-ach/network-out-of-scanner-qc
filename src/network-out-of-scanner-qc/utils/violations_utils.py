import pandas as pd
import os
from pathlib import Path
import re
import numpy as np
from utils.utils import filter_to_test_trials, sort_subject_ids
import matplotlib.pyplot as plt
import seaborn as sns

def check_violation_conditions(current_trial, next_valid_trial):
    return (current_trial['stop_signal_condition'] == 'go' and
            next_valid_trial['stop_signal_condition'] == 'stop' and
            current_trial['rt'] != -1 and
            next_valid_trial['rt'] != -1)

def find_difference(stop_rt, go_rt):
    return stop_rt - go_rt

def get_ssd(next_valid_trial):
    return next_valid_trial['SS_delay']

def compute_violations(subject_id, df, task_name):
    violations_row = []

    df = filter_to_test_trials(df, task_name)

    for i in range(len(df) - 1): 
        current_trial = df.iloc[i]
        # Check for a Go trial followed by a Stop trial with a violation
        if current_trial['stop_signal_condition'] == 'go':
            next_trials = df.iloc[i+1:]
            next_valid_trial = next_trials[next_trials['stop_signal_condition'].notna()].iloc[0] if not next_trials.empty else None

            if next_valid_trial is not None and next_valid_trial['stop_signal_condition'] == 'stop':
                if check_violation_conditions(current_trial, next_valid_trial):
                    go_rt = current_trial['rt']
                    stop_rt = next_valid_trial['rt']
                    ssd = get_ssd(next_valid_trial)
                    difference = find_difference(stop_rt, go_rt)
                    violations_row.append({'subject_id': subject_id, 'task_name': task_name, 'ssd': ssd, 'difference': difference, 'violation': go_rt < stop_rt})

    return pd.DataFrame(violations_row)

def aggregate_violations(violations_df):
    aggregated_violations_df = violations_df.groupby(['subject_id', 'task_name', 'ssd']).agg(
        difference_mean=('difference', 'mean'),
        proportion_violation=('violation', 'mean'),
    ).reset_index()
    aggregated_violations_df = sort_subject_ids(aggregated_violations_df)
    return aggregated_violations_df

def plot_violations(aggregated_violations_df, violations_output_path):
    # Get unique subjects and tasks
    subjects = aggregated_violations_df['subject_id'].unique()
    tasks = sorted(aggregated_violations_df['task_name'].unique())

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=len(subjects), ncols=len(tasks), 
                             figsize=(5*len(tasks), 3*len(subjects)), 
                             squeeze=False)

    # Set common y-limit for all plots
    y_min = aggregated_violations_df['difference_mean'].min()
    y_max = aggregated_violations_df['difference_mean'].max()
    y_range = y_max - y_min
    y_limit = (y_min - 0.1*y_range, y_max + 0.1*y_range)

    # Set common x-limit for all plots
    x_min = aggregated_violations_df['ssd'].min()
    x_max = aggregated_violations_df['ssd'].max()
    x_range = x_max - x_min
    x_limit = (x_min - 0.1*x_range, x_max + 0.1*x_range)

    for i, subject in enumerate(subjects):
        for j, task in enumerate(tasks):
            ax = axes[i, j]
            
            # Filter data for this subject and task
            data = aggregated_violations_df[(aggregated_violations_df['subject_id'] == subject) & 
                                            (aggregated_violations_df['task_name'] == task)]
            
            if not data.empty:
                ax.scatter(data['ssd'], data['difference_mean'], color='blue')
                ax.axhline(0, color='red', linestyle='--')
                ax.set_ylim(y_limit)
                ax.set_xlim(x_limit)  # Set the same x-limits for all plots
                
                # Only set x and y labels for the leftmost and bottom subplots
                if j == 0:
                    ax.set_ylabel('Avg Stop RT - Go RT')
                if i == len(subjects) - 1:
                    ax.set_xlabel('SSD (ms)')
                
                # Remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            
            # Set title only for the top row with larger font
            if i == 0:
                ax.set_title(task, fontsize=14)
            
            # Set subject ID only for the leftmost column with larger font
            if j == 0:
                ax.text(-0.5, 0.5, subject, transform=ax.transAxes, 
                        ha='right', va='center', fontsize=14)

    # Adjust layout
    plt.tight_layout()
    
    # Add extra space on the left for subject IDs
    plt.subplots_adjust(left=0.15)

    # Save the figure
    plt.savefig(violations_output_path / 'violations_matrix.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_violations_matrices(aggregated_violations_df, violations_output_path):
    for task in aggregated_violations_df['task_name'].unique():
        task_df = aggregated_violations_df[aggregated_violations_df['task_name'] == task]
        task_df = task_df.pivot(index='subject_id', columns='ssd', values='proportion_violation')
        #create mean row and column
        task_df.loc['mean'] = task_df.mean(axis=0)
        task_df.loc[:, 'mean'] = task_df.mean(axis=1)
        task_df.to_csv(violations_output_path / f'{task}_violations_matrix.csv')
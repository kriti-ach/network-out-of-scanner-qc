import pandas as pd
import os
from pathlib import Path
import re
import numpy as np
from utils.utils import filter_to_test_trials, sort_subject_ids
import matplotlib.pyplot as plt
import seaborn as sns

def compute_violations(subject_id, df, task_name):
    violations_row = []
    delay = 1 if task_name == 'stop_signal_with_n_back' else 2

    df = filter_to_test_trials(df, task_name)

    for i in range(len(df) - delay): 
        # Check for a Go trial followed by a Stop trial with a violation
        if (df.iloc[i]['stop_signal_condition'] == 'go' and
            df.iloc[i + delay]['stop_signal_condition'] == 'stop' and
            (df.iloc[i + delay]['rt'] != -1)):
            
            go_rt = df.iloc[i]['rt']         # RT of Go trial
            stop_rt = df.iloc[i + delay]['rt']  # RT of Stop trial
            
            if pd.notna(go_rt) and pd.notna(stop_rt):  # Ensure RTs are valid
                difference = stop_rt - go_rt  # Calculate the difference
                ssd = df.iloc[i + delay]['SS_delay']    # SSD for the Stop trial
                violations_row.append({'subject_id': subject_id, 'task_name': task_name, 'ssd': ssd, 'difference': difference})

    return pd.DataFrame(violations_row)

def aggregate_violations(violations_df):
    aggregated_violations_df = violations_df.groupby(['subject_id', 'task_name', 'ssd']).agg(
        difference_mean=('difference', 'mean'),
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

    # Set a common y-limit for all plots
    y_min = aggregated_violations_df['difference_mean'].min()
    y_max = aggregated_violations_df['difference_mean'].max()
    print(f'y_min: {y_min}, y_max: {y_max}')
    y_range = y_max - y_min
    y_limit = (y_min - 0.1*y_range, y_max + 0.1*y_range)

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
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
    STOP_SIGNAL_CONDITIONS,
    GO_NOGO_CONDITIONS,
    SHAPE_MATCHING_CONDITIONS,
    FLANKER_WITH_CUED_CONDITIONS,
    GO_NOGO_WITH_CUED_CONDITIONS,
    SHAPE_MATCHING_WITH_CUED_CONDITIONS,
    CUED_TASK_SWITCHING_WITH_DIRECTED_FORGETTING_CONDITIONS,
    SHAPE_MATCHING_CONDITIONS_WITH_DIRECTED_FORGETTING
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

def get_dual_n_back_columns(base_columns, sample_df, paired_col=None, cuedts=False):
    """
    Generate columns for dual n-back tasks (n-back paired with another task).
    - base_columns: list of base columns (e.g., ['subject_id'])
    - sample_df: DataFrame with sample data
    - paired_col: column name for the paired task (e.g., 'go_nogo_condition', 'flanker_condition')
    - cuedts: if True, handle n-back with cued task switching
    Returns: list of columns
    """
    conditions = []
    if sample_df is not None:
        if cuedts:
            cue_conditions = [c for c in sample_df['cue_condition'].unique() if pd.notna(c) and str(c).lower() != 'na']
            task_conditions = [t for t in sample_df['task_condition'].unique() if pd.notna(t) and str(t).lower() != 'na']
            for n_back_condition in sample_df['n_back_condition'].unique():
                if pd.isna(n_back_condition):
                    continue
                for delay in sample_df['delay'].unique():
                    if pd.isna(delay):
                        continue
                    for cue in cue_conditions:
                        for taskc in task_conditions:
                            if cue == "stay" and taskc == "switch":
                                continue
                            else:
                                col_prefix = f"{n_back_condition}_{delay}back_t{taskc}_c{cue}"
                                conditions.append(col_prefix)
            return extend_metric_columns(base_columns, conditions)
        else:
            conditions = [
                f"{n_back_condition}_{delay}back_{paired_condition}"
                for n_back_condition in sample_df['n_back_condition'].unique()
                for delay in sample_df['delay'].unique()
                for paired_condition in sample_df[paired_col].unique()
            ]
        return extend_metric_columns(base_columns, conditions)
    return base_columns  # Return base columns if no sample data available

def get_task_columns(task_name, sample_df=None):
    """
    Define columns for each task's QC CSV.
    """
    base_columns = ['subject_id']
    
    if is_dual_task(task_name):
        if 'directed_forgetting' in task_name and 'flanker' in task_name or 'directedForgetting' in task_name and 'flanker' in task_name:
            # For dual tasks, create combined condition names
            conditions = [
                f'{df_cond}_{f_cond}'
                for df_cond in DIRECTED_FORGETTING_CONDITIONS
                for f_cond in FLANKER_CONDITIONS
            ]
            return extend_metric_columns(base_columns, conditions)
        elif 'flanker' in task_name and 'shape_matching' in task_name:
            conditions = [
                f'{f_cond}_{s_cond}'
                for f_cond in FLANKER_CONDITIONS
                for s_cond in SHAPE_MATCHING_CONDITIONS
            ]
            return extend_metric_columns(base_columns, conditions)
        elif 'directed_forgetting' in task_name and 'go_nogo' in task_name:
            conditions = [
                f'{df_cond}_{g_cond}'
                for df_cond in DIRECTED_FORGETTING_CONDITIONS
                for g_cond in GO_NOGO_CONDITIONS
            ]
            return extend_metric_columns(base_columns, conditions)
        elif 'directed_forgetting' in task_name and 'shape_matching' in task_name or 'directedForgetting' in task_name and 'shape_matching' in task_name:
            conditions = [
                f'{df_cond}_{s_cond}'
                for df_cond in DIRECTED_FORGETTING_CONDITIONS
                for s_cond in SHAPE_MATCHING_CONDITIONS_WITH_DIRECTED_FORGETTING
            ]
            return extend_metric_columns(base_columns, conditions)
        elif 'flanker' in task_name and 'go_nogo' in task_name:
            conditions = [
                f'{f_cond}_{g_cond}'
                for f_cond in FLANKER_CONDITIONS
                for g_cond in GO_NOGO_CONDITIONS
            ]
            return extend_metric_columns(base_columns, conditions)
        elif 'go_nogo' in task_name and 'shape_matching' in task_name:
            conditions = [
                f'{g_cond}_{s_cond}'
                for g_cond in GO_NOGO_CONDITIONS
                for s_cond in SHAPE_MATCHING_CONDITIONS
            ]
            return extend_metric_columns(base_columns, conditions)
        elif 'directed_forgetting' in task_name and 'spatial_task_switching' in task_name or 'directedForgetting' in task_name and 'spatialTS' in task_name:
            conditions = [
                f'{df_cond}_{s_cond}'
                for df_cond in DIRECTED_FORGETTING_CONDITIONS
                for s_cond in SPATIAL_TASK_SWITCHING_CONDITIONS
            ]
            return extend_metric_columns(base_columns, conditions)
        elif 'cued_task_switching' in task_name and 'spatial_task_switching' in task_name or 'CuedTS' in task_name and 'spatialTS' in task_name:
            return extend_metric_columns(base_columns, SPATIAL_WITH_CUED_CONDITIONS)
        elif 'flanker' in task_name and 'cued_task_switching' in task_name or 'flanker' in task_name and 'CuedTS' in task_name:
            return extend_metric_columns(base_columns, FLANKER_WITH_CUED_CONDITIONS)
        elif 'go_nogo' in task_name and 'cued_task_switching' in task_name or 'go_nogo' in task_name and 'CuedTS' in task_name:
            return extend_metric_columns(base_columns, GO_NOGO_WITH_CUED_CONDITIONS)
        elif 'shape_matching' in task_name and 'cued_task_switching' in task_name or 'shape_matching' in task_name and 'CuedTS' in task_name:
            return extend_metric_columns(base_columns, SHAPE_MATCHING_WITH_CUED_CONDITIONS)
        elif 'directed_forgetting' in task_name and 'cued_task_switching' in task_name or 'directedForgetting' in task_name and 'CuedTS' in task_name:
            return extend_metric_columns(base_columns, CUED_TASK_SWITCHING_WITH_DIRECTED_FORGETTING_CONDITIONS)
        elif 'go_nogo' in task_name and 'n_back' in task_name or 'go_nogo' in task_name and 'NBack' in task_name:
            return get_dual_n_back_columns(base_columns, sample_df, 'go_nogo_condition')
        elif 'flanker' in task_name and 'n_back' in task_name or 'flanker' in task_name and 'NBack' in task_name:
            return get_dual_n_back_columns(base_columns, sample_df, 'flanker_condition')
        elif 'shape_matching' in task_name and 'n_back' in task_name or 'shape_matching' in task_name and 'NBack' in task_name:
            return get_dual_n_back_columns(base_columns, sample_df, 'shape_matching_condition')
        elif 'directed_forgetting' in task_name and 'n_back' in task_name or 'directedForgetting' in task_name and 'NBack' in task_name:
            return get_dual_n_back_columns(base_columns, sample_df, 'directed_forgetting_condition')
        elif 'n_back' in task_name and 'cued_task_switching' in task_name or 'NBack' in task_name and 'CuedTS' in task_name:
            return get_dual_n_back_columns(base_columns, sample_df, cuedts=True)
    else:
        if 'spatial_task_switching' in task_name or 'spatialTS' in task_name:
            return extend_metric_columns(base_columns, SPATIAL_TASK_SWITCHING_CONDITIONS)
        elif 'cued_task_switching' in task_name or 'cuedTS' in task_name:
            return extend_metric_columns(base_columns, CUED_TASK_SWITCHING_CONDITIONS)
        elif 'directed_forgetting' in task_name or 'directedForgetting' in task_name:
            return extend_metric_columns(base_columns, DIRECTED_FORGETTING_CONDITIONS)
        elif 'flanker' in task_name:
            return extend_metric_columns(base_columns, FLANKER_CONDITIONS)
        elif 'n_back' in task_name:
            # For n-back, we need to get the columns from the data
            if sample_df is not None:
                conditions = [
                    f"{n_back_condition}_{delay}back"
                    for n_back_condition in sample_df['n_back_condition'].unique()
                    for delay in sample_df['delay'].unique()
                ]
                return extend_metric_columns(base_columns, conditions)
            return base_columns  # Return base columns if no sample data available
        elif 'stop_signal' in task_name:
            columns = [
                'subject_id',
                'go_rt',
                'stop_fail_rt',
                'go_acc',
                'stop_failure_acc',
                'stop_success',
                'avg_ssd',
                'min_ssd',
                'max_ssd',
                'min_ssd_count',
                'max_ssd_count',
                'ssrt'
            ]
            return columns
        elif 'go_nogo' in task_name:
            return extend_metric_columns(base_columns, GO_NOGO_CONDITIONS)
        elif 'shape_matching' in task_name:
            return extend_metric_columns(base_columns, SHAPE_MATCHING_CONDITIONS)
        else:
            print(f"Unknown task: {task_name}")
            return None

def is_dual_task(task_name):
    """
    Check if the task is a dual task.
    """
    return any(task in task_name for task in DUAL_TASKS_OUT_OF_SCANNER)
    # return any(task in task_name for task in DUAL_TASKS_FMRI)

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
    qc_file = output_path / f"{task_name}_qc.csv"
    try:
        df = pd.read_csv(qc_file)
        # Add any new columns from metrics that aren't in the DataFrame
        for key in metrics.keys():
            if key not in df.columns:
                df[key] = np.nan
        new_row = pd.DataFrame({
            'subject_id': [subject_id],
            **metrics
        })
        df = pd.concat([df, new_row], ignore_index=True)
        if task_name == 'flanker_with_cued_task_switching' or task_name == 'shape_matching_with_cued_task_switching':
            df = df.drop(columns=[col for col in df.columns if 'tswitch_new_c' in col])
        df.to_csv(qc_file, index=False)
    except FileNotFoundError:
        print(f"Warning: QC file {qc_file} not found")

def compute_cued_task_switching_metrics(
    df,
    condition_list,
    condition_type,
    flanker_col=None,
    go_nogo_col=None,
    shape_matching_col=None,
    directed_forgetting_col=None
):
    """
    Compute metrics for cued task switching and its duals (flanker/go_nogo).
    - condition_list: list of condition strings (e.g., FLANKER_WITH_CUED_CONDITIONS, GO_NOGO_WITH_CUED_CONDITIONS, or CUED_TASK_SWITCHING_CONDITIONS)
    - condition_type: 'single', 'flanker', or 'go_nogo'
    - flanker_col: column name for flanker (if dual)
    - go_nogo_col: column name for go_nogo (if dual)
    """
    metrics = {}
    for cond in condition_list:
        try:
            if condition_type == 'single':
                # cond format: t{task}_c{cue}
                if not cond.startswith('t') or '_c' not in cond:
                    continue
                task = cond[1:cond.index('_c')]
                cue = cond[cond.index('_c')+2:]
                mask_acc = (df['task_condition'].apply(lambda x: str(x).lower()) == task) & \
                           (df['cue_condition'].apply(lambda x: str(x).lower()) == cue)
            elif condition_type == 'flanker':
                # cond format: {flanker}_t{task}_c{cue}
                flanker, t_part = cond.split('_t')
                task, cue = t_part.split('_c')
                mask_acc = (
                    df[flanker_col].str.contains(flanker, case=False, na=False) &
                    (df['task_condition'].apply(lambda x: str(x).lower()) == ('switch' if task in ['switch', 'switch_new'] else task)) &
                    (df['cue_condition'].apply(lambda x: str(x).lower()) == cue)
                )
            elif condition_type == 'go_nogo':
                # cond format: {go_nogo}_t{task}_c{cue}
                go_nogo, t_part = cond.split('_t')
                task, cue = t_part.split('_c')
                mask_acc = (
                    (df[go_nogo_col].apply(lambda x: str(x).lower()) == go_nogo) &
                    (df['task_condition'].apply(lambda x: str(x).lower()) == task) &
                    (df['cue_condition'].apply(lambda x: str(x).lower()) == cue)
                )
            elif condition_type == 'shape_matching':
                # cond format: {shape_matching}_t{task}_c{cue}
                shape_matching, t_part = cond.split('_t')
                task, cue = t_part.split('_c')
                mask_acc = (
                    (df[shape_matching_col].apply(lambda x: str(x)) == shape_matching) &
                    (df['task_condition'].apply(lambda x: str(x).lower()) == ('switch' if task in ['switch', 'switch_new'] else task)) &
                    (df['cue_condition'].apply(lambda x: str(x).lower()) == cue)
                )
            elif condition_type == 'directed_forgetting':
                # cond format: {directed_forgetting}_t{task}_c{cue}
                directed_forgetting, t_part = cond.split('_t')
                task, cue = t_part.split('_c')
                mask_acc = (
                    (df[directed_forgetting_col].apply(lambda x: str(x).lower()) == directed_forgetting) &
                    (df['task_condition'].apply(lambda x: str(x).lower()) == task) &
                    (df['cue_condition'].apply(lambda x: str(x).lower()) == cue)
                )
            else:
                continue
        except Exception as e:
            print(f"Skipping malformed condition: {cond} ({e})")
            continue
        mask_rt = mask_acc & (df['correct_trial'] == 1)
        mask_omission = mask_acc & (df['key_press'] == -1)
        mask_commission = mask_acc & (df['key_press'] != -1) & (df['correct_trial'] == 0)
        num_omissions = len(df[mask_omission])
        num_commissions = len(df[mask_commission])
        total_num_trials = len(df[mask_acc])
        metrics[f'{cond}_acc'] = df[mask_acc]['correct_trial'].mean()
        metrics[f'{cond}_rt'] = df[mask_rt]['rt'].mean()
        metrics[f'{cond}_omission_rate'] = num_omissions / total_num_trials if total_num_trials > 0 else np.nan
        metrics[f'{cond}_commission_rate'] = num_commissions / total_num_trials if total_num_trials > 0 else np.nan
    return metrics

def compute_n_back_metrics(df, condition_list, paired_task_col=None, paired_conditions=None, cuedts=False):
    """
    Compute metrics for n-back tasks (single, dual, or n-back with cuedts).
    - df: DataFrame
    - condition_list: list of n-back conditions (e.g., ['0', '2']) or list of tuples for duals
    - paired_task_col: column name for the paired task (if dual)
    - paired_conditions: list of paired task conditions (if dual)
    - cuedts: if True, handle n-back with cued task switching
    Returns: dict of metrics
    """
    metrics = {}
    if cuedts:
        cue_conditions = [c for c in df['cue_condition'].unique() if pd.notna(c) and str(c).lower() != 'na']
        task_conditions = [t for t in df['task_condition'].unique() if pd.notna(t) and str(t).lower() != 'na']
        for n_back_condition in df['n_back_condition'].unique():
            if pd.isna(n_back_condition):
                continue
            for delay in df['delay'].unique():
                if pd.isna(delay):
                    continue
                for cue in cue_conditions:
                    for taskc in task_conditions:
                        if cue == "stay" and taskc == "switch":
                            continue
                        col_prefix = f"{n_back_condition}_{delay}back_t{taskc}_c{cue}"
                        mask_acc = (
                            (df['n_back_condition'] == n_back_condition) &
                            (df['delay'] == delay) &
                            (df['cue_condition'] == cue) &
                            (df['task_condition'] == taskc)
                        )
                        mask_rt = mask_acc & (df['correct_trial'] == 1)
                        mask_omission = mask_acc & (df['key_press'] == -1)
                        mask_commission = mask_acc & (df['key_press'] != -1) & (df['correct_trial'] == 0)
                        num_omissions = len(df[mask_omission])
                        num_commissions = len(df[mask_commission])
                        total_num_trials = len(df[mask_acc])
                        metrics[f'{col_prefix}_acc'] = df[mask_acc]['correct_trial'].mean()
                        metrics[f'{col_prefix}_rt'] = df[mask_rt]['rt'].mean()
                        metrics[f'{col_prefix}_omission_rate'] = num_omissions / total_num_trials if total_num_trials > 0 else np.nan
                        metrics[f'{col_prefix}_commission_rate'] = num_commissions / total_num_trials if total_num_trials > 0 else np.nan
        return metrics
    if paired_task_col is None:
        # Single n-back: iterate over n_back_condition and delay
        for n_back_condition in df['n_back_condition'].unique():
            if pd.isna(n_back_condition):
                continue
            for delay in df['delay'].unique():
                if pd.isna(delay):
                    continue
                condition = f"{n_back_condition}_{delay}back"
                mask_acc = (df['n_back_condition'] == n_back_condition) & (df['delay'] == delay)
                mask_rt = mask_acc & (df['correct_trial'] == 1)
                metrics[f'{condition}_acc'] = df[mask_acc]['correct_trial'].mean()
                metrics[f'{condition}_rt'] = df[mask_rt]['rt'].mean()
    else:
        # Dual n-back: iterate over n_back_condition, delay, and paired task conditions
        for n_back_condition in df['n_back_condition'].unique():
            if pd.isna(n_back_condition):
                continue
            for delay in df['delay'].unique():
                if pd.isna(delay):
                    continue
                for paired_cond in paired_conditions:
                    condition = f"{n_back_condition}_{delay}back_{paired_cond}"
                    mask_acc = (df['n_back_condition'] == n_back_condition) & (df['delay'] == delay) & (df[paired_task_col] == paired_cond)
                    mask_rt = mask_acc & (df['correct_trial'] == 1)
                    metrics[f'{condition}_acc'] = df[mask_acc]['correct_trial'].mean()
                    metrics[f'{condition}_rt'] = df[mask_rt]['rt'].mean()
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
        
        elif ('directed_forgetting' in task_name and 'go_nogo' in task_name) or ('directedForgetting' in task_name and 'go_nogo' in task_name):
            conditions = {
                'directed_forgetting': DIRECTED_FORGETTING_CONDITIONS,
                'go_nogo': GO_NOGO_CONDITIONS
            }
            condition_columns = {
                'directed_forgetting': 'directed_forgetting_condition',
                'go_nogo': 'go_nogo_condition'
            }
            return calculate_metrics(df, conditions, condition_columns, is_dual_task(task_name))
        
        elif ('flanker' in task_name and 'go_nogo' in task_name) or ('flanker' in task_name and 'go_nogo' in task_name):
            conditions = {
                'flanker': FLANKER_CONDITIONS,
                'go_nogo': GO_NOGO_CONDITIONS
            }
            condition_columns = {
                'flanker': 'flanker_condition',
                'go_nogo': 'go_nogo_condition'
            }
            return calculate_metrics(df, conditions, condition_columns, is_dual_task(task_name))
        
        elif ('directed_forgetting' in task_name and 'shape_matching' in task_name) or ('directedForgetting' in task_name and 'shape_matching' in task_name):
            conditions = {
                'directed_forgetting': DIRECTED_FORGETTING_CONDITIONS,
                'shape_matching': SHAPE_MATCHING_CONDITIONS_WITH_DIRECTED_FORGETTING
            }
            condition_columns = {
                'directed_forgetting': 'directed_forgetting_condition',
                'shape_matching': 'shape_matching_condition'
            }
            return calculate_metrics(df, conditions, condition_columns, is_dual_task(task_name))
        
        elif ('go_nogo' in task_name and 'shape_matching' in task_name) or ('go_nogo' in task_name and 'shape_matching' in task_name):
            conditions = {
                'go_nogo': GO_NOGO_CONDITIONS,
                'shape_matching': SHAPE_MATCHING_CONDITIONS
            }
            condition_columns = {
                'go_nogo': 'go_nogo_condition',
                'shape_matching': 'shape_matching_condition'
            }
            return calculate_metrics(df, conditions, condition_columns, is_dual_task(task_name))
        
        elif ('flanker' in task_name and 'shape_matching' in task_name) or ('flanker' in task_name and 'shape_matching' in task_name):
            conditions = {
                'flanker': FLANKER_CONDITIONS,
                'shape_matching': SHAPE_MATCHING_CONDITIONS
            }
            condition_columns = {
                'flanker': 'flanker_condition',
                'shape_matching': 'shape_matching_condition'
            }
            return calculate_metrics(df, conditions, condition_columns, is_dual_task(task_name))
        
        elif ('spatial_task_switching' in task_name and 'directed_forgetting' in task_name) or ('spatialTS' in task_name and 'directedForgetting' in task_name):
            conditions = {
                'spatial_task_switching': SPATIAL_TASK_SWITCHING_CONDITIONS,
                'directed_forgetting': DIRECTED_FORGETTING_CONDITIONS
            }
            condition_columns = {
                'spatial_task_switching': 'task_switch',
                'directed_forgetting': 'directed_forgetting_condition'
            }
            return calculate_metrics(df, conditions, condition_columns, is_dual_task(task_name))
        
        elif ('cued_task_switching' in task_name and 'spatial_task_switching' in task_name) or ('CuedTS' in task_name and 'spatialTS' in task_name):
            metrics = {}
            for cond in SPATIAL_WITH_CUED_CONDITIONS:
                # Parse condition like 'cuedtstaycstay_spatialtstaycstay'
                try:
                    cued_part, spatial_part = cond.split('_spatial')
                    # Extract task and cue from cued part (e.g., 'cuedtstaycstay' -> 'stay' and 'stay')
                    t_start = cued_part.index('t')
                    # Find the 'c' that starts the cue part (after the task)
                    if 'cstay' in cued_part:
                        c_start = cued_part.index('cstay')
                        cued_task = cued_part[t_start + 1:c_start]  # Extract 'stay' from 'cuedtstaycstay'
                        cued_cue = 'stay'
                    elif 'cswitch' in cued_part:
                        c_start = cued_part.index('cswitch')
                        cued_task = cued_part[t_start + 1:c_start]  # Extract 'switch' from 'cuedtswitchcswitch'
                        cued_cue = 'switch'
                    else:
                        # Fallback: find 'c' after 't'
                        c_start = cued_part.index('c', t_start)
                        cued_task = cued_part[t_start + 1:c_start]
                        cued_cue = cued_part[c_start + 1:]
                    
                    # Extract task and cue from spatial part (e.g., 'tswitchcswitch' -> 'switch' and 'switch')
                    if 'cstay' in spatial_part:
                        c_start = spatial_part.index('cstay')
                        spatial_task = spatial_part[1:c_start]  # Extract 'stay' from 'tstaycstay'
                        spatial_cue = 'stay'
                    elif 'cswitch' in spatial_part:
                        c_start = spatial_part.index('cswitch')
                        spatial_task = spatial_part[1:c_start]  # Extract 'switch' from 'tswitchcswitch'
                        spatial_cue = 'switch'
                    else:
                        # Fallback: find 'c' after 't'
                        c_start = spatial_part.index('c')
                        spatial_task = spatial_part[1:c_start]
                        spatial_cue = spatial_part[c_start + 1:]
                    # Create mask for both cued and spatial parts
                    mask_acc = (
                        (df['cue_condition'] == cued_cue) & 
                        (df['task_condition'] == cued_task) & 
                        (df['task_switch'] == f't{spatial_task}_c{spatial_cue}')
                    )
                    mask_rt = mask_acc & (df['correct_trial'] == 1)
                    mask_omission = mask_acc & (df['key_press'] == -1)
                    mask_commission = mask_acc & (df['key_press'] != -1) & (df['correct_trial'] == 0)
                    num_omissions = len(df[mask_omission])
                    num_commissions = len(df[mask_commission])
                    total_num_trials = len(df[mask_acc])
                    metrics[f'{cond}_acc'] = df[mask_acc]['correct_trial'].mean()
                    metrics[f'{cond}_rt'] = df[mask_rt]['rt'].mean()
                    metrics[f'{cond}_omission_rate'] = num_omissions / total_num_trials if total_num_trials > 0 else np.nan
                    metrics[f'{cond}_commission_rate'] = num_commissions / total_num_trials if total_num_trials > 0 else np.nan
                except Exception as e:
                    print(f"Error parsing condition {cond}: {e}")
                    continue
            return metrics
        elif ('flanker' in task_name and 'cued_task_switching' in task_name) or ('flanker' in task_name and 'CuedTS' in task_name):
            return compute_cued_task_switching_metrics(df, FLANKER_WITH_CUED_CONDITIONS, 'flanker', flanker_col='flanker_condition')
        elif ('go_nogo' in task_name and 'cued_task_switching' in task_name) or ('go_nogo' in task_name and 'CuedTS' in task_name):
            return compute_cued_task_switching_metrics(df, GO_NOGO_WITH_CUED_CONDITIONS, 'go_nogo', go_nogo_col='go_nogo_condition')
        elif ('shape_matching' in task_name and 'cued_task_switching' in task_name) or ('shape_matching' in task_name and 'CuedTS' in task_name):
            return compute_cued_task_switching_metrics(df, SHAPE_MATCHING_WITH_CUED_CONDITIONS, 'shape_matching', shape_matching_col='shape_matching_condition')
        elif ('directed_forgetting' in task_name and 'cued_task_switching' in task_name) or ('directedForgetting' in task_name and 'CuedTS' in task_name):
            return compute_cued_task_switching_metrics(df, CUED_TASK_SWITCHING_WITH_DIRECTED_FORGETTING_CONDITIONS, 'directed_forgetting', directed_forgetting_col='directed_forgetting_condition')
        elif ('n_back' in task_name and 'go_nogo' in task_name) or ('NBack' in task_name and 'go_nogo' in task_name):
            # Example: dual n-back with go_nogo
            paired_conditions = [c for c in df['go_nogo_condition'].unique() if pd.notna(c)]
            return compute_n_back_metrics(df, None, paired_task_col='go_nogo_condition', paired_conditions=paired_conditions)
        elif ('n_back' in task_name and 'flanker' in task_name) or ('NBack' in task_name and 'flanker' in task_name):
            paired_conditions = [c for c in df['flanker_condition'].unique() if pd.notna(c)]
            return compute_n_back_metrics(df, None, paired_task_col='flanker_condition', paired_conditions=paired_conditions)
        elif ('n_back' in task_name and 'shape_matching' in task_name) or ('NBack' in task_name and 'shape_matching' in task_name):
            paired_conditions = [c for c in df['shape_matching_condition'].unique() if pd.notna(c)]
            return compute_n_back_metrics(df, None, paired_task_col='shape_matching_condition', paired_conditions=paired_conditions)
        elif ('n_back' in task_name and 'directed_forgetting' in task_name) or ('NBack' in task_name and 'directed_forgetting' in task_name):
            paired_conditions = [c for c in df['directed_forgetting_condition'].unique() if pd.notna(c)]
            return compute_n_back_metrics(df, None, paired_task_col='directed_forgetting_condition', paired_conditions=paired_conditions)
        elif ('n_back' in task_name and 'cued_task_switching' in task_name) or ('NBack' in task_name and 'CuedTS' in task_name):
            return compute_n_back_metrics(df, None, paired_task_col='task_switch', paired_conditions=None, cuedts=True)
        # Add more dual n-back pairings as needed
    else:
        # Special handling for n-back task
        if 'n_back' in task_name:
            return compute_n_back_metrics(df, None)

        elif 'cued_task_switching' in task_name:
            return compute_cued_task_switching_metrics(df, CUED_TASK_SWITCHING_CONDITIONS, 'single')
        # Special handling for stop signal task
        elif 'stop_signal' in task_name:
            metrics = {}
            # Use your column names
            go_mask = (df['SS_trial_type'] == 'go')
            stop_mask = (df['SS_trial_type'] == 'stop')
            stop_fail_mask = stop_mask & (df['correct_trial'] == 0)
            stop_succ_mask = stop_mask & (df['correct_trial'] == 1)

            # RTs
            metrics['go_rt'] = df.loc[go_mask & (df['rt'].notna()), 'rt'].mean()
            metrics['stop_fail_rt'] = df.loc[stop_fail_mask & (df['rt'].notna()), 'rt'].mean()

            # Accuracies
            metrics['go_acc'] = df.loc[go_mask, 'correct_trial'].mean()
            # 1. Learn mapping from go trials
            go_trials = df[go_mask]
            stim_to_resp = (
                go_trials.groupby('stim')['correct_response']
                .agg(lambda x: x.value_counts().idxmax())
                .to_dict()
            )

            # 2. For stop-failure trials with a response
            stop_fail_with_resp = df[stop_fail_mask & (df['key_press'] != -1)]
            if not stop_fail_with_resp.empty:
                stop_fail_with_resp = stop_fail_with_resp.copy()
                stop_fail_with_resp['expected_response'] = stop_fail_with_resp['stim'].map(stim_to_resp)
                stop_fail_with_resp['is_correct'] = stop_fail_with_resp['key_press'] == stop_fail_with_resp['expected_response']
                metrics['stop_failure_acc'] = stop_fail_with_resp['is_correct'].mean()
            else:
                metrics['stop_failure_acc'] = np.nan

            metrics['stop_success'] = len(df[stop_succ_mask])/len(df[stop_mask])
            # SSD stats
            ssd_vals = df.loc[stop_mask, 'SS_delay'].dropna()
            metrics['avg_ssd'] = ssd_vals.mean()
            metrics['min_ssd'] = ssd_vals.min()
            metrics['max_ssd'] = ssd_vals.max()
            metrics['min_ssd_count'] = (ssd_vals == metrics['min_ssd']).sum() if not np.isnan(metrics['min_ssd']) else 0
            metrics['max_ssd_count'] = (ssd_vals == metrics['max_ssd']).sum() if not np.isnan(metrics['max_ssd']) else 0

            # SSRT
            metrics['ssrt'] = compute_SSRT(df)

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
            condition_columns = {'spatial_task_switching': 'task_switch'}
        elif 'go_nogo' in task_name:    
            conditions = {'go_nogo': GO_NOGO_CONDITIONS}
            condition_columns = {'go_nogo': 'go_nogo_condition'}
        elif 'shape_matching' in task_name:
            conditions = {'shape_matching': SHAPE_MATCHING_CONDITIONS}
            condition_columns = {'shape_matching': 'shape_matching_condition'}
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
                mask_acc = df[condition_columns[task1]].str.contains(cond1, case=False, na=False) & \
                           df[condition_columns[task2]].str.contains(cond2, case=False, na=False)
                mask_rt = mask_acc & (df['correct_trial'] == 1)
                metrics[f'{cond1}_{cond2}_acc'] = df[mask_acc]['correct_trial'].mean()
                metrics[f'{cond1}_{cond2}_rt'] = df[mask_rt]['rt'].mean()
                mask_omission = mask_acc & (df['key_press'] == -1)
                mask_commission = mask_acc & (df['key_press'] != -1) & (df['correct_trial'] == 0)
                num_omissions = len(df[mask_omission])
                num_commissions = len(df[mask_commission])
                total_num_trials = len(df[mask_acc])
                metrics[f'{cond1}_{cond2}_omission_rate'] = num_omissions / total_num_trials
                metrics[f'{cond1}_{cond2}_commission_rate'] = num_commissions / total_num_trials
    else:
        # For single tasks, just iterate through conditions
        task = list(conditions.keys())[0]
        for cond in conditions[task]:
            mask_acc = (df[condition_columns[task]] == cond)
            mask_rt = (df[condition_columns[task]] == cond) & (df['correct_trial'] == 1)
            metrics[f'{cond}_acc'] = df[mask_acc]['correct_trial'].mean()
            metrics[f'{cond}_rt'] = df[mask_rt]['rt'].mean()
            mask_omission = (df[condition_columns[task]] == cond) & (df['key_press'] == -1)
            mask_commission = (df[condition_columns[task]] == cond) & (df['key_press'] != -1) & (df['correct_trial'] == 0)
            num_omissions = len(df[mask_omission])
            num_commissions = len(df[mask_commission])
            total_num_trials = len(df[mask_acc])
            metrics[f'{cond}_omission_rate'] = num_omissions / total_num_trials
            metrics[f'{cond}_commission_rate'] = num_commissions / total_num_trials
    
    return metrics

def append_summary_rows_to_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return
    if df.empty or len(df.columns) < 2:
        return

    stats_cols = df.columns[1:]  # Assuming first column is subject_id
    summary = {}
    for stat, func in zip(['mean', 'std', 'max', 'min'], [np.mean, np.std, np.max, np.min]):
        row = []
        for col in df.columns:
            if col in stats_cols:
                val = func(df[col]) if pd.api.types.is_numeric_dtype(df[col]) else np.nan
                row.append(val)
            else:
                row.append(stat)
        summary[stat] = row

    for stat, values in summary.items():
        df.loc[len(df)] = values
    df.to_csv(csv_path, index=False)

def compute_SSRT(df, max_go_rt=2):
    # Only use test phase if present
    if 'Phase' in df.columns:
        df = df.query('Phase == "test"')
    go_trials = df[df['SS_trial_type'] == 'go']
    stop_df = df[df['SS_trial_type'] == 'stop']

    go_replacement_df = go_trials.copy()
    go_replacement_df['rt'] = go_replacement_df['rt'].fillna(max_go_rt)
    sorted_go = go_replacement_df['rt'].sort_values(ascending=True, ignore_index=True)
    stop_failure = stop_df[stop_df['rt'].notna()]
    if len(stop_df) > 0:
        p_respond = len(stop_failure) / len(stop_df)
        avg_SSD = stop_df['SS_delay'].mean()
    else:
        return np.nan

    nth_index = int(np.rint(p_respond * len(sorted_go))) - 1
    if nth_index < 0:
        nth_RT = sorted_go.iloc[0]
    elif nth_index >= len(sorted_go):
        nth_RT = sorted_go.iloc[-1]
    else:
        nth_RT = sorted_go.iloc[nth_index]

    if avg_SSD is not None and not np.isnan(avg_SSD):
        SSRT = nth_RT - avg_SSD
    else:
        SSRT = np.nan

    return SSRT
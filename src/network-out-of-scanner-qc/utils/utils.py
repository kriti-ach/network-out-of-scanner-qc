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

def extend_go_nogo_metric_columns(base_columns, conditions):
    """
    Extend base columns with accuracy and RT metrics for go_nogo tasks.
    For nogo conditions: only acc and rt (no omission/commission rates)
    For go conditions: all metrics (acc, rt, omission_rate, commission_rate)
    
    Args:
        base_columns (list): Base columns (e.g., ['subject_id'])
        conditions (list): List of go_nogo conditions (e.g., ['go', 'nogo'] or ['flanker_go', 'flanker_nogo'])
        
    Returns:
        list: Extended list of columns with appropriate metrics for each condition
    """
    columns = base_columns.copy()
    for cond in conditions:
        # Check if this condition ends with 'nogo' or contains '_nogo_'
        if cond.endswith('_nogo') or '_nogo_' in cond or cond == 'nogo':
            # For nogo: only acc and rt
            columns.extend([f'{cond}_acc', f'{cond}_rt'])
        else:
            # For go: all metrics
            columns.extend([f'{cond}_acc', f'{cond}_rt', f'{cond}_omission_rate', f'{cond}_commission_rate'])
    return columns

def get_dual_n_back_columns(base_columns, sample_df, paired_col=None, cuedts=False, gonogo=False):
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
        elif gonogo:
            for n_back_condition in sample_df['n_back_condition'].str.lower().unique():
                for delay in sample_df['delay'].unique():
                    for paired_condition in sample_df[paired_col].str.lower().unique():
                        if paired_condition == 'nogo':
                            conditions.extend([
                                f"{n_back_condition}_{delay}back_{paired_condition}_acc",
                                f"{n_back_condition}_{delay}back_{paired_condition}_rt"
                            ])
                        else:
                            conditions.extend([
                                f"{n_back_condition}_{delay}back_{paired_condition}_acc",
                                f"{n_back_condition}_{delay}back_{paired_condition}_rt",
                                f"{n_back_condition}_{delay}back_{paired_condition}_omission_rate",
                                f"{n_back_condition}_{delay}back_{paired_condition}_commission_rate"
                            ])
            return extend_go_nogo_metric_columns(base_columns, conditions)
        else:
            conditions = [
                f"{n_back_condition}_{delay}back_{paired_condition}"
                for n_back_condition in sample_df['n_back_condition'].str.lower().unique()
                for delay in sample_df['delay'].unique()
                for paired_condition in sample_df[paired_col].str.lower().unique()
            ]
        return extend_metric_columns(base_columns, conditions)
    return base_columns  # Return base columns if no sample data available

def create_dual_task_conditions(task1_conditions, task2_conditions, separator='_'):
    """
    Create combined condition names for dual tasks.
    
    Args:
        task1_conditions (list): Conditions for first task
        task2_conditions (list): Conditions for second task
        separator (str): Separator between task conditions
        
    Returns:
        list: Combined condition names
    """
    return [f'{cond1}{separator}{cond2}' for cond1 in task1_conditions for cond2 in task2_conditions]

def create_stop_signal_dual_columns(paired_conditions, include_nogo_commission=False):
    """
    Create column names for stop signal dual tasks.
    
    Args:
        paired_conditions (list): List of paired task conditions
        include_nogo_commission (bool): Whether to include nogo_commission_rate column
        
    Returns:
        list: Column names for stop signal dual task
    """
    conditions = []
    for condition in paired_conditions:
        conditions.extend([
            f"{condition}_go_rt",
            f"{condition}_stop_fail_rt",
            f"{condition}_go_acc",
            f"{condition}_stop_fail_acc",
            f"{condition}_stop_success",
            f"{condition}_go_omission_rate",
            f"{condition}_go_commission_rate",
            f"{condition}_ssrt"  # Add SSRT column for each condition
        ])
    
    if include_nogo_commission:
        conditions.append("nogo_commission_rate")
    
    return conditions

def get_task_columns(task_name, sample_df=None):
    """
    Define columns for each task's QC CSV.
    """
    base_columns = ['subject_id']
    
    if is_dual_task(task_name):
        # Handle dual tasks with stop signal first
        if ('stop_signal' in task_name or 'stopSignal' in task_name) and 'flanker' in task_name:
            if sample_df is not None:
                flanker_conditions = [c for c in sample_df['flanker_condition'].unique() if pd.notna(c)]
                return base_columns + create_stop_signal_dual_columns(flanker_conditions)
            return base_columns
        elif ('stop_signal' in task_name or 'stopSignal' in task_name) and 'go_nogo' in task_name:
            if sample_df is not None:
                go_conditions = [c for c in sample_df['go_nogo_condition'].unique() if pd.notna(c) and c == 'go']
                return base_columns + create_stop_signal_dual_columns(go_conditions, include_nogo_commission=True)
            return base_columns
        elif ('stop_signal' in task_name or 'stopSignal' in task_name) and 'shape_matching' in task_name:
            if sample_df is not None:
                shape_conditions = [c for c in sample_df['shape_matching_condition'].unique() if pd.notna(c)]
                return base_columns + create_stop_signal_dual_columns(shape_conditions)
            return base_columns
        elif ('stop_signal' in task_name or 'stopSignal' in task_name) and 'directed_forgetting' in task_name:
            if sample_df is not None:
                df_conditions = [c for c in sample_df['directed_forgetting_condition'].unique() if pd.notna(c)]
                return base_columns + create_stop_signal_dual_columns(df_conditions)
            return base_columns
        elif ('stop_signal' in task_name or 'stopSignal' in task_name) and 'spatial_task_switching' in task_name:
            if sample_df is not None:
                spatial_conditions = [c for c in sample_df['task_switch'].unique() if pd.notna(c) and c != 'na']
                return base_columns + create_stop_signal_dual_columns(spatial_conditions)
            return base_columns
        elif ('stop_signal' in task_name or 'stopSignal' in task_name) and 'cued_task_switching' in task_name:
            if sample_df is not None:
                conditions = []
                for cue_condition in sample_df['cue_condition'].unique():
                    if pd.notna(cue_condition) and str(cue_condition).lower() != 'na':
                        for task_condition in sample_df['task_condition'].unique():
                            if pd.notna(task_condition) and str(task_condition).lower() != 'na':
                                if cue_condition == "stay" and task_condition == "switch":
                                    continue
                                conditions.append(f"t{task_condition}_c{cue_condition}")
                return base_columns + create_stop_signal_dual_columns(conditions)
            return base_columns
        elif ('stop_signal' in task_name or 'stopSignal' in task_name) and 'n_back' in task_name:
            if sample_df is not None:
                conditions = []
                for n_back_condition in sample_df['n_back_condition'].unique():
                    if pd.notna(n_back_condition):
                        for delay in sample_df['delay'].unique():
                            if pd.notna(delay):
                                condition_name = f"{n_back_condition}_{delay}back"
                                conditions.append(condition_name)
                return base_columns + create_stop_signal_dual_columns(conditions)
            return base_columns
        
        # Handle regular dual tasks (non-stop signal)
        elif 'directed_forgetting' in task_name and 'flanker' in task_name or 'directedForgetting' in task_name and 'flanker' in task_name:
            conditions = create_dual_task_conditions(DIRECTED_FORGETTING_CONDITIONS, FLANKER_CONDITIONS)
            return extend_metric_columns(base_columns, conditions)
        elif 'flanker' in task_name and 'shape_matching' in task_name:
            conditions = create_dual_task_conditions(FLANKER_CONDITIONS, SHAPE_MATCHING_CONDITIONS)
            return extend_metric_columns(base_columns, conditions)
        elif 'flanker' in task_name and 'go_nogo' in task_name:
            conditions = create_dual_task_conditions(FLANKER_CONDITIONS, GO_NOGO_CONDITIONS)
            return extend_go_nogo_metric_columns(base_columns, conditions)
        elif 'directed_forgetting' in task_name and 'go_nogo' in task_name:
            conditions = create_dual_task_conditions(DIRECTED_FORGETTING_CONDITIONS, GO_NOGO_CONDITIONS)
            return extend_go_nogo_metric_columns(base_columns, conditions)
        elif 'directed_forgetting' in task_name and 'shape_matching' in task_name or 'directedForgetting' in task_name and 'shape_matching' in task_name:
            conditions = create_dual_task_conditions(DIRECTED_FORGETTING_CONDITIONS, SHAPE_MATCHING_CONDITIONS_WITH_DIRECTED_FORGETTING)
            return extend_metric_columns(base_columns, conditions)
        elif 'go_nogo' in task_name and 'shape_matching' in task_name:
            conditions = create_dual_task_conditions(GO_NOGO_CONDITIONS, SHAPE_MATCHING_CONDITIONS)
            return extend_go_nogo_metric_columns(base_columns, conditions)
        elif 'directed_forgetting' in task_name and 'spatial_task_switching' in task_name or 'directedForgetting' in task_name and 'spatialTS' in task_name:
            conditions = create_dual_task_conditions(SPATIAL_TASK_SWITCHING_CONDITIONS, DIRECTED_FORGETTING_CONDITIONS)
            return extend_metric_columns(base_columns, conditions)
        elif 'flanker' in task_name and 'spatial_task_switching' in task_name or 'flanker' in task_name and 'spatialTS' in task_name:
            conditions = create_dual_task_conditions(SPATIAL_TASK_SWITCHING_CONDITIONS, FLANKER_CONDITIONS)
            return extend_metric_columns(base_columns, conditions)
        elif 'go_nogo' in task_name and 'spatial_task_switching' in task_name or 'go_nogo' in task_name and 'spatialTS' in task_name:
            conditions = create_dual_task_conditions(SPATIAL_TASK_SWITCHING_CONDITIONS, GO_NOGO_CONDITIONS)
            return extend_go_nogo_metric_columns(base_columns, conditions)
        elif 'shape_matching' in task_name and 'spatial_task_switching' in task_name or 'shape_matching' in task_name and 'spatialTS' in task_name:
            conditions = create_dual_task_conditions(SPATIAL_TASK_SWITCHING_CONDITIONS, SHAPE_MATCHING_CONDITIONS)
            return extend_metric_columns(base_columns, conditions)
        elif 'cued_task_switching' in task_name and 'spatial_task_switching' in task_name or 'CuedTS' in task_name and 'spatialTS' in task_name:
            return extend_metric_columns(base_columns, SPATIAL_WITH_CUED_CONDITIONS)
        elif 'flanker' in task_name and 'cued_task_switching' in task_name or 'flanker' in task_name and 'CuedTS' in task_name:
            return extend_metric_columns(base_columns, FLANKER_WITH_CUED_CONDITIONS)
        elif 'go_nogo' in task_name and 'cued_task_switching' in task_name or 'go_nogo' in task_name and 'CuedTS' in task_name:
            return extend_go_nogo_metric_columns(base_columns, GO_NOGO_WITH_CUED_CONDITIONS)
        elif 'shape_matching' in task_name and 'cued_task_switching' in task_name or 'shape_matching' in task_name and 'CuedTS' in task_name:
            return extend_metric_columns(base_columns, SHAPE_MATCHING_WITH_CUED_CONDITIONS)
        elif 'directed_forgetting' in task_name and 'cued_task_switching' in task_name or 'directedForgetting' in task_name and 'CuedTS' in task_name:
            return extend_metric_columns(base_columns, CUED_TASK_SWITCHING_WITH_DIRECTED_FORGETTING_CONDITIONS)
        elif 'go_nogo' in task_name and 'n_back' in task_name or 'go_nogo' in task_name and 'NBack' in task_name:
            return get_dual_n_back_columns(base_columns, sample_df, 'go_nogo_condition', gonogo=True)
        elif 'flanker' in task_name and 'n_back' in task_name or 'flanker' in task_name and 'NBack' in task_name:
            return get_dual_n_back_columns(base_columns, sample_df, 'flanker_condition')
        elif 'shape_matching' in task_name and 'n_back' in task_name or 'shape_matching' in task_name and 'NBack' in task_name:
            return get_dual_n_back_columns(base_columns, sample_df, 'shape_matching_condition')
        elif 'directed_forgetting' in task_name and 'n_back' in task_name or 'directedForgetting' in task_name and 'NBack' in task_name:
            return get_dual_n_back_columns(base_columns, sample_df, 'directed_forgetting_condition')
        elif 'n_back' in task_name and 'cued_task_switching' in task_name or 'NBack' in task_name and 'CuedTS' in task_name:
            return get_dual_n_back_columns(base_columns, sample_df, cuedts=True)
        elif 'n_back' in task_name and 'spatial_task_switching' in task_name or 'NBack' in task_name and 'spatialTS' in task_name:
            return get_dual_n_back_columns(base_columns, sample_df, 'task_switch')
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
                'stop_fail_acc',
                'stop_success',
                'go_omission_rate',
                'go_commission_rate',
                'avg_ssd',
                'min_ssd',
                'max_ssd',
                'min_ssd_count',
                'max_ssd_count',
                'ssrt'
            ]
            return columns
        elif 'go_nogo' in task_name:
            return extend_go_nogo_metric_columns(base_columns, GO_NOGO_CONDITIONS)
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
        # Ensure new_row has the same columns as df
        for col in df.columns:
            if col not in new_row.columns:
                new_row[col] = np.nan
        # Reorder columns to match df
        new_row = new_row[df.columns]
        
        # Ensure data types match
        for col in df.columns:
            if col in new_row.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    new_row[col] = pd.to_numeric(new_row[col], errors='coerce')
                elif pd.api.types.is_string_dtype(df[col]):
                    new_row[col] = new_row[col].astype(str)
        
        df = pd.concat([df, new_row], ignore_index=True)
        if task_name == 'flanker_with_cued_task_switching' or task_name == 'shape_matching_with_cued_task_switching':
            df = df.drop(columns=[col for col in df.columns if 'tswitch_new_c' in col])
        df['subject_id_numeric'] = df['subject_id'].str.replace('s', '').astype(int)
        # Sort the DataFrame
        df = df.sort_values(by='subject_id_numeric', ascending=True)

        # Remove 'subject_id_numeric' and add the 's' back to 'subject_id'
        df['subject_id'] = 's' + df['subject_id_numeric'].astype(str)
        df = df.drop(columns=['subject_id_numeric'])
        df.to_csv(qc_file, index=False)
    except FileNotFoundError:
        print(f"Warning: QC file {qc_file} not found")

def calculate_accuracy(df, mask_acc):
    """
    Calculate accuracy for given mask.
    
    Args:
        df (pd.DataFrame): DataFrame containing task data
        mask_acc (pd.Series): Boolean mask for accuracy calculation
        
    Returns:
        float: Accuracy (mean of correct_trial)
    """
    return df[mask_acc]['correct_trial'].mean() if len(df[mask_acc]) > 0 else np.nan

def calculate_rt(df, mask_rt):
    """
    Calculate reaction time for given mask.
    
    Args:
        df (pd.DataFrame): DataFrame containing task data
        mask_rt (pd.Series): Boolean mask for RT calculation (correct trials only)
        
    Returns:
        float: Mean reaction time
    """
    return df[mask_rt]['rt'].mean() if len(df[mask_rt]) > 0 else np.nan

def calculate_omission_rate(df, mask_omission, total_num_trials):
    """
    Calculate omission rate for given mask.
    
    Args:
        df (pd.DataFrame): DataFrame containing task data
        mask_omission (pd.Series): Boolean mask for omission calculation
        total_num_trials (int): Total number of trials for normalization
        
    Returns:
        float: Omission rate
    """
    num_omissions = len(df[mask_omission])
    return num_omissions / total_num_trials if total_num_trials > 0 else np.nan

def calculate_commission_rate(df, mask_commission, total_num_trials):
    """
    Calculate commission rate for given mask.
    
    Args:
        df (pd.DataFrame): DataFrame containing task data
        mask_commission (pd.Series): Boolean mask for commission calculation
        total_num_trials (int): Total number of trials for normalization
        
    Returns:
        float: Commission rate
    """
    num_commissions = len(df[mask_commission])
    return num_commissions / total_num_trials if total_num_trials > 0 else np.nan

def calculate_basic_metrics(df, mask_acc, cond_name, metrics_dict):
    """
    Calculate all basic metrics (accuracy, RT, omission rate, commission rate) for a condition.
    
    Args:
        df (pd.DataFrame): DataFrame containing task data
        mask_acc (pd.Series): Boolean mask for accuracy calculation
        cond_name (str): Condition name for metric keys
        metrics_dict (dict): Dictionary to store metrics
        
    Returns:
        None: Updates metrics_dict in place
    """
    mask_rt = mask_acc & (df['correct_trial'] == 1)
    mask_omission = mask_acc & (df['key_press'] == -1)
    mask_commission = mask_acc & (df['key_press'] != -1) & (df['correct_trial'] == 0)
    total_num_trials = len(df[mask_acc])
    
    metrics_dict[f'{cond_name}_acc'] = calculate_accuracy(df, mask_acc)
    metrics_dict[f'{cond_name}_rt'] = calculate_rt(df, mask_rt)
    metrics_dict[f'{cond_name}_omission_rate'] = calculate_omission_rate(df, mask_omission, total_num_trials)
    metrics_dict[f'{cond_name}_commission_rate'] = calculate_commission_rate(df, mask_commission, total_num_trials)

def calculate_go_nogo_metrics(df, mask_acc, cond_name, metrics_dict):
    """
    Calculate go_nogo metrics with special handling for nogo condition.
    For nogo: only calculate RT for commission errors, no omission/commission rates.
    For go: calculate all metrics normally.
    
    Args:
        df (pd.DataFrame): DataFrame containing task data
        mask_acc (pd.Series): Boolean mask for accuracy calculation
        cond_name (str): Condition name for metric keys
        metrics_dict (dict): Dictionary to store metrics
        
    Returns:
        None: Updates metrics_dict in place
    """
    # Check if this is a nogo condition
    is_nogo = cond_name.endswith('_nogo') or '_nogo_' in cond_name or cond_name == 'nogo'
    
    if is_nogo:
        # For nogo: only calculate RT for commission errors (incorrect responses)
        mask_rt = mask_acc & (df['key_press'] != -1) & (df['correct_trial'] == 0)
        metrics_dict[f'{cond_name}_acc'] = calculate_accuracy(df, mask_acc)
        metrics_dict[f'{cond_name}_rt'] = calculate_rt(df, mask_rt)
        # Don't calculate omission_rate or commission_rate for nogo
    else:
        # For go: calculate all metrics normally
        mask_rt = mask_acc & (df['correct_trial'] == 1)
        mask_omission = mask_acc & (df['key_press'] == -1)
        mask_commission = mask_acc & (df['key_press'] != -1) & (df['correct_trial'] == 0)
        total_num_trials = len(df[mask_acc])
        
        metrics_dict[f'{cond_name}_acc'] = calculate_accuracy(df, mask_acc)
        metrics_dict[f'{cond_name}_rt'] = calculate_rt(df, mask_rt)
        metrics_dict[f'{cond_name}_omission_rate'] = calculate_omission_rate(df, mask_omission, total_num_trials)
        metrics_dict[f'{cond_name}_commission_rate'] = calculate_commission_rate(df, mask_commission, total_num_trials)

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
                calculate_basic_metrics(df, mask_acc, cond, metrics)
            elif condition_type == 'flanker':
                # cond format: {flanker}_t{task}_c{cue}
                flanker, t_part = cond.split('_t')
                task, cue = t_part.split('_c')
                mask_acc = (
                    df[flanker_col].str.contains(flanker, case=False, na=False) &
                    (df['task_condition'].apply(lambda x: str(x).lower()) == ('switch' if task in ['switch', 'switch_new'] else task)) &
                    (df['cue_condition'].apply(lambda x: str(x).lower()) == cue)
                )
                calculate_basic_metrics(df, mask_acc, cond, metrics)
            elif condition_type == 'go_nogo':
                # cond format: {go_nogo}_t{task}_c{cue}
                go_nogo, t_part = cond.split('_t')
                task, cue = t_part.split('_c')
                mask_acc = (
                    (df[go_nogo_col].apply(lambda x: str(x).lower()) == go_nogo) &
                    (df['task_condition'].apply(lambda x: str(x).lower()) == task) &
                    (df['cue_condition'].apply(lambda x: str(x).lower()) == cue)
                )
                calculate_go_nogo_metrics(df, mask_acc, cond, metrics)
            elif condition_type == 'shape_matching':
                # cond format: {shape_matching}_t{task}_c{cue}
                shape_matching, t_part = cond.split('_t')
                task, cue = t_part.split('_c')
                mask_acc = (
                    (df[shape_matching_col].apply(lambda x: str(x)) == shape_matching) &
                    (df['task_condition'].apply(lambda x: str(x).lower()) == ('switch' if task in ['switch', 'switch_new'] else task)) &
                    (df['cue_condition'].apply(lambda x: str(x).lower()) == cue)
                )
                calculate_basic_metrics(df, mask_acc, cond, metrics)
            elif condition_type == 'directed_forgetting':
                # cond format: {directed_forgetting}_t{task}_c{cue}
                directed_forgetting, t_part = cond.split('_t')
                task, cue = t_part.split('_c')
                mask_acc = (
                    (df[directed_forgetting_col].apply(lambda x: str(x).lower()) == directed_forgetting) &
                    (df['task_condition'].apply(lambda x: str(x).lower()) == task) &
                    (df['cue_condition'].apply(lambda x: str(x).lower()) == cue)
                )
                calculate_basic_metrics(df, mask_acc, cond, metrics)
        except Exception as e:
            print(f"Skipping malformed condition: {cond} ({e})")
            continue
    return metrics

def compute_n_back_metrics(df, condition_list, paired_task_col=None, paired_conditions=None, cuedts=False, gonogo=False):
    """
    Compute metrics for n-back tasks (single, dual, or n-back with cuedts).
    - df: DataFrame
    - condition_list: list of n-back conditions (e.g., ['0', '2']) or list of tuples for duals
    - paired_task_col: column name for the paired task (if dual)
    - paired_conditions: list of paired task conditions (if dual)
    - cuedts: if True, handle n-back with cued task switching
    - gonogo: if True, handle n-back with go_nogo task switching
    Returns: dict of metrics
    """
    metrics = {}
    if cuedts:
        cue_conditions = [c for c in df['cue_condition'].unique() if pd.notna(c) and str(c).lower() != 'na']
        task_conditions = [t for t in df['task_condition'].unique() if pd.notna(t) and str(t).lower() != 'na']
        for n_back_condition in df['n_back_condition'].str.lower().unique():
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
                            (df['n_back_condition'].str.lower() == n_back_condition) &
                            (df['delay'] == delay) &
                            (df['cue_condition'] == cue) &
                            (df['task_condition'] == taskc)
                        )
                        calculate_basic_metrics(df, mask_acc, col_prefix, metrics)
        return metrics
    elif gonogo:
        for n_back_condition in df['n_back_condition'].str.lower().unique():
            if pd.isna(n_back_condition):
                continue
            for delay in df['delay'].unique():
                for paired_condition in paired_conditions:
                    col_prefix = f"{n_back_condition}_{delay}back_{paired_condition}"
                    mask_acc = (df['n_back_condition'].str.lower() == n_back_condition) & (df['delay'] == delay) & (df[paired_task_col].str.lower() == paired_condition.lower())
                    calculate_go_nogo_metrics(df, mask_acc, col_prefix, metrics)
        return metrics
    if paired_task_col is None:
        # Single n-back: iterate over n_back_condition and delay
        for n_back_condition in df['n_back_condition'].str.lower().unique():
            if pd.isna(n_back_condition):
                continue
            for delay in df['delay'].unique():
                if pd.isna(delay):
                    continue
                condition = f"{n_back_condition}_{delay}back"
                mask_acc = (df['n_back_condition'].str.lower() == n_back_condition) & (df['delay'] == delay)
                calculate_basic_metrics(df, mask_acc, condition, metrics)
    else:
        # Dual n-back: iterate over n_back_condition, delay, and paired task conditions
        for n_back_condition in df['n_back_condition'].str.lower().unique():
            if pd.isna(n_back_condition):
                continue
            for delay in df['delay'].unique():
                if pd.isna(delay):
                    continue
                for paired_cond in paired_conditions:
                    condition = f"{n_back_condition}_{delay}back_{paired_cond.lower()}"
                    mask_acc = (df['n_back_condition'].str.lower() == n_back_condition) & (df['delay'] == delay) & (df[paired_task_col].str.lower() == paired_cond.lower())
                    calculate_basic_metrics(df, mask_acc, condition, metrics)
    return metrics

def compute_cued_spatial_task_switching_metrics(df, condition_list):
    """
    Compute metrics for cued task switching with spatial task switching dual task.
    
    Args:
        df (pd.DataFrame): DataFrame containing task data
        condition_list (list): List of combined conditions (e.g., SPATIAL_WITH_CUED_CONDITIONS)
        
    Returns:
        dict: Metrics for cued + spatial task switching
    """
    metrics = {}
    for cond in condition_list:
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
            calculate_basic_metrics(df, mask_acc, cond, metrics)
        except Exception as e:
            print(f"Error parsing condition {cond}: {e}")
            continue
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
        
        elif ('spatial_task_switching' in task_name and 'flanker' in task_name) or ('spatialTS' in task_name and 'flanker' in task_name):
            conditions = {
                'spatial_task_switching': SPATIAL_TASK_SWITCHING_CONDITIONS,
                'flanker': FLANKER_CONDITIONS
            }
            condition_columns = {
                'spatial_task_switching': 'task_switch',
                'flanker': 'flanker_condition'
            }
            return calculate_metrics(df, conditions, condition_columns, is_dual_task(task_name))
        
        elif ('spatial_task_switching' in task_name and 'go_nogo' in task_name) or ('spatialTS' in task_name and 'go_nogo' in task_name):
            conditions = {
                'spatial_task_switching': SPATIAL_TASK_SWITCHING_CONDITIONS,
                'go_nogo': GO_NOGO_CONDITIONS
            }
            condition_columns = {
                'spatial_task_switching': 'task_switch',
                'go_nogo': 'go_nogo_condition'
            }
            return calculate_metrics(df, conditions, condition_columns, is_dual_task(task_name))
        
        elif ('spatial_task_switching' in task_name and 'shape_matching' in task_name) or ('spatialTS' in task_name and 'shape_matching' in task_name):
            conditions = {
                'spatial_task_switching': SPATIAL_TASK_SWITCHING_CONDITIONS,
                'shape_matching': SHAPE_MATCHING_CONDITIONS
            }
            condition_columns = {
                'spatial_task_switching': 'task_switch',
                'shape_matching': 'shape_matching_condition'
            }
            return calculate_metrics(df, conditions, condition_columns, is_dual_task(task_name))
        
        elif ('cued_task_switching' in task_name and 'spatial_task_switching' in task_name) or ('CuedTS' in task_name and 'spatialTS' in task_name):
            return compute_cued_spatial_task_switching_metrics(df, SPATIAL_WITH_CUED_CONDITIONS)
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
            return compute_n_back_metrics(df, None, paired_task_col='go_nogo_condition', paired_conditions=paired_conditions, gonogo=True)
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
        elif ('n_back' in task_name and 'spatial_task_switching' in task_name) or ('NBack' in task_name and 'spatialTS' in task_name):
            paired_conditions = [c for c in df['task_switch'].unique() if pd.notna(c) and c != 'na']
            return compute_n_back_metrics(df, None, paired_task_col='task_switch', paired_conditions=paired_conditions)
        elif ('stop_signal' in task_name and 'flanker' in task_name) or ('stopSignal' in task_name and 'flanker' in task_name):
            paired_conditions = [c for c in df['flanker_condition'].unique() if pd.notna(c)]
            return compute_stop_signal_metrics(df, dual_task=True, paired_task_col='flanker_condition', paired_conditions=paired_conditions, stim_col='center_letter')
        elif ('stop_signal' in task_name and 'go_nogo' in task_name) or ('stopSignal' in task_name and 'go_nogo' in task_name):
            # Only process 'go' condition, not 'nogo'
            paired_conditions = ['go']  # Only process go condition
            metrics = compute_stop_signal_metrics(df, dual_task=True, paired_task_col='go_nogo_condition', paired_conditions=paired_conditions, stim_col='go_nogo_condition')
            
            # Calculate nogo commission rate separately
            nogo_mask = (df['go_nogo_condition'] == 'nogo')
            nogo_commission_mask = nogo_mask & (df['key_press'] != -1) & (df['correct_trial'] == 0)
            num_nogo_commissions = len(df[nogo_commission_mask])
            total_nogo_trials = len(df[nogo_mask])
            metrics['nogo_commission_rate'] = num_nogo_commissions / total_nogo_trials if total_nogo_trials > 0 else np.nan
            
            return metrics
        elif ('stop_signal' in task_name and 'shape_matching' in task_name) or ('stopSignal' in task_name and 'shape_matching' in task_name):
            paired_conditions = [c for c in df['shape_matching_condition'].unique() if pd.notna(c)]
            return compute_stop_signal_metrics(df, dual_task=True, paired_task_col='shape_matching_condition', paired_conditions=paired_conditions, stim_col='shape_matching_condition')
        elif ('stop_signal' in task_name and 'directed_forgetting' in task_name) or ('stopSignal' in task_name and 'directedForgetting' in task_name):
            paired_conditions = [c for c in df['directed_forgetting_condition'].unique() if pd.notna(c)]
            return compute_stop_signal_metrics(df, dual_task=True, paired_task_col='directed_forgetting_condition', paired_conditions=paired_conditions, stim_col='directed_forgetting_condition')
        elif ('stop_signal' in task_name and 'spatial_task_switching' in task_name) or ('stopSignal' in task_name and 'spatialTS' in task_name):
            paired_conditions = [c for c in df['task_switch'].unique() if pd.notna(c) and c != 'na']
            return compute_stop_signal_metrics(df, dual_task=True, paired_task_col='task_switch', paired_conditions=paired_conditions, stim_cols=['number', 'predictable_dimension'])
        elif ('stop_signal' in task_name and 'n_back' in task_name) or ('stopSignal' in task_name and 'NBack' in task_name):
            paired_conditions = []
            for n_back_condition in df['n_back_condition'].unique():
                if pd.notna(n_back_condition):
                    for delay in df['delay'].unique():
                        if pd.notna(delay):
                            paired_conditions.append(f"{n_back_condition}_{delay}back")
            return compute_stop_signal_metrics(df, dual_task=True, paired_task_col=None, paired_conditions=paired_conditions, stim_col='n_back_condition')
        elif ('stop_signal' in task_name and 'cued_task_switching' in task_name) or ('stopSignal' in task_name and 'CuedTS' in task_name):
            # Create combined conditions for cued task switching (e.g., "tstay_cstay", "tstay_cswitch")
            paired_conditions = []
            for cue_condition in df['cue_condition'].unique():
                if pd.notna(cue_condition) and str(cue_condition).lower() != 'na':
                    for task_condition in df['task_condition'].unique():
                        if pd.notna(task_condition) and str(task_condition).lower() != 'na':
                            # Skip the combination where cue is stay and task is switch
                            if cue_condition == "stay" and task_condition == "switch":
                                continue
                            paired_conditions.append(f"t{task_condition}_c{cue_condition}")
            return compute_stop_signal_metrics(df, dual_task=True, paired_task_col=None, paired_conditions=paired_conditions, stim_cols=['stim_number', 'task'])
        # Add more dual n-back pairings as needed
    else:
        # Special handling for n-back task
        if 'n_back' in task_name:
            return compute_n_back_metrics(df, None)

        elif 'cued_task_switching' in task_name:
            return compute_cued_task_switching_metrics(df, CUED_TASK_SWITCHING_CONDITIONS, 'single')
        # Special handling for stop signal task
        elif 'stop_signal' in task_name:
            return compute_stop_signal_metrics(df, dual_task=False)
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
                # Check if this is a go_nogo task
                if 'go_nogo' in task1 or 'go_nogo' in task2:
                    calculate_go_nogo_metrics(df, mask_acc, f'{cond1}_{cond2}', metrics)
                else:
                    calculate_basic_metrics(df, mask_acc, f'{cond1}_{cond2}', metrics)
    else:
        # For single tasks, just iterate through conditions
        task = list(conditions.keys())[0]
        for cond in conditions[task]:
            mask_acc = (df[condition_columns[task]] == cond)
            # Check if this is a go_nogo task
            if 'go_nogo' in task:
                calculate_go_nogo_metrics(df, mask_acc, cond, metrics)
            else:
                calculate_basic_metrics(df, mask_acc, cond, metrics)
    
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

def calculate_single_stop_signal_metrics(df):
    """
    Calculate metrics for single stop signal task.
    
    Args:
        df (pd.DataFrame): DataFrame containing task data
        
    Returns:
        dict: Metrics for single stop signal task
    """
    metrics = {}
    
    go_mask = (df['SS_trial_type'] == 'go')
    stop_mask = (df['SS_trial_type'] == 'stop')
    stop_fail_mask = stop_mask & (df['correct_trial'] == 0)
    stop_succ_mask = stop_mask & (df['correct_trial'] == 1)

    # RTs
    metrics['go_rt'] = df.loc[go_mask & (df['rt'].notna()) & (df['rt'] > 0), 'rt'].mean()
    metrics['stop_fail_rt'] = df.loc[stop_fail_mask & (df['rt'].notna()) & (df['rt'] > 0), 'rt'].mean()

    # Accuracies
    metrics['go_acc'] = df.loc[go_mask, 'correct_trial'].mean()
    
    # Stop failure accuracy based on stimulus-response mapping from go trials
    go_trials = df[go_mask]
    stim_to_resp = (
        go_trials.groupby('stim')['correct_response']
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )

    # Go omission rate
    go_mask = (df['SS_trial_type'] == 'go')
    mask_omission = go_mask & (df['key_press'] == -1)
    mask_commission = go_mask & (df['key_press'] != -1) & (df['correct_trial'] == 0)
    metrics['go_omission_rate'] = calculate_omission_rate(df, mask_omission, len(df[go_mask]))
    metrics['go_commission_rate'] = calculate_commission_rate(df, mask_commission, len(df[go_mask]))

    stop_fail_with_resp = df[stop_fail_mask & (df['key_press'] != -1)]
    if not stop_fail_with_resp.empty:
        stop_fail_with_resp = stop_fail_with_resp.copy()
        stop_fail_with_resp['expected_response'] = stop_fail_with_resp['stim'].map(stim_to_resp)
        stop_fail_with_resp['is_correct'] = stop_fail_with_resp['key_press'] == stop_fail_with_resp['expected_response']
        metrics['stop_fail_acc'] = stop_fail_with_resp['is_correct'].mean()
    else:
        metrics['stop_fail_acc'] = np.nan

    metrics['stop_success'] = len(df[stop_succ_mask])/len(df[stop_mask])
    
    return metrics

def calculate_stop_signal_ssd_stats(df):
    """
    Calculate SSD statistics for stop signal task.
    
    Args:
        df (pd.DataFrame): DataFrame containing task data
        
    Returns:
        dict: SSD statistics
    """
    stop_mask = (df['SS_trial_type'] == 'stop')
    ssd_vals = df.loc[stop_mask, 'SS_delay'].dropna()
    
    metrics = {}
    metrics['avg_ssd'] = ssd_vals.mean()
    metrics['min_ssd'] = ssd_vals.min()
    metrics['max_ssd'] = ssd_vals.max()
    metrics['min_ssd_count'] = (ssd_vals == metrics['min_ssd']).sum() if not np.isnan(metrics['min_ssd']) else 0
    metrics['max_ssd_count'] = (ssd_vals == metrics['max_ssd']).sum() if not np.isnan(metrics['max_ssd']) else 0
    
    return metrics

def calculate_dual_stop_signal_condition_metrics(df, paired_cond, paired_mask, stim_col=None, stim_cols=None):
    """
    Calculate stop signal metrics for a single condition in dual task.
    
    Args:
        df (pd.DataFrame): DataFrame containing task data
        paired_cond (str): Paired task condition name
        paired_mask (pd.Series): Boolean mask for the condition
        stim_col (str): Single stimulus column for mapping
        stim_cols (list): Multiple stimulus columns for mapping
        
    Returns:
        dict: Metrics for the condition
    """
    metrics = {}
    
    go_mask = (df['SS_trial_type'] == 'go') & paired_mask
    stop_mask = (df['SS_trial_type'] == 'stop') & paired_mask
    stop_fail_mask = stop_mask & (df['key_press'] != -1)
    stop_succ_mask = stop_mask & (df['key_press'] == -1)

    # RTs
    metrics[f'{paired_cond}_go_rt'] = df.loc[go_mask & (df['rt'].notna()) & (df['rt'] > 0), 'rt'].mean()
    metrics[f'{paired_cond}_stop_fail_rt'] = df.loc[stop_fail_mask & (df['rt'].notna()) & (df['rt'] > 0), 'rt'].mean()

    # Accuracies
    go_trials = df[go_mask]
    correct_go_trials = (go_trials['key_press'] == go_trials['correct_response']).sum()
    metrics[f'{paired_cond}_go_acc'] = correct_go_trials / len(go_trials)
    # Go omission rate
    mask_omission = go_mask & (df['key_press'] == -1)
    mask_commission = go_mask & (df['key_press'] != -1) & (df['key_press'] != df['correct_response'])
    metrics[f'{paired_cond}_go_omission_rate'] = calculate_omission_rate(df, mask_omission, len(go_trials))
    metrics[f'{paired_cond}_go_commission_rate'] = calculate_commission_rate(df, mask_commission, len(go_trials))
    
    # Stop failure accuracy based on stimulus-response mapping from go trials
    if stim_col is not None:
        if not go_trials.empty:
            stim_to_resp = (
                go_trials.groupby(stim_col)['correct_response']
                .agg(lambda x: x.value_counts().idxmax())
                .to_dict()
            )

            stop_fail_with_resp = df[stop_fail_mask & (df['key_press'] != -1)]
            if not stop_fail_with_resp.empty:
                stop_fail_with_resp = stop_fail_with_resp.copy()
                stop_fail_with_resp['expected_response'] = stop_fail_with_resp[stim_col].map(stim_to_resp)
                stop_fail_with_resp['is_correct'] = stop_fail_with_resp['key_press'] == stop_fail_with_resp['expected_response']
                metrics[f'{paired_cond}_stop_fail_acc'] = stop_fail_with_resp['is_correct'].mean()
            else:
                metrics[f'{paired_cond}_stop_fail_acc'] = np.nan
        else:
            metrics[f'{paired_cond}_stop_fail_acc'] = np.nan
    elif stim_cols is not None:
        if not go_trials.empty:
            # Group by multiple stimulus columns
            stim_to_resp = (
                go_trials.groupby(stim_cols)['correct_response']
                .agg(lambda x: x.value_counts().idxmax())
                .to_dict()
            )

            stop_fail_with_resp = df[stop_fail_mask & (df['key_press'] != -1)]
            if not stop_fail_with_resp.empty:
                stop_fail_with_resp = stop_fail_with_resp.copy()
                # Create a tuple key from multiple stimulus columns
                stop_fail_with_resp['stim_key'] = stop_fail_with_resp[stim_cols].apply(tuple, axis=1)
                stop_fail_with_resp['expected_response'] = stop_fail_with_resp['stim_key'].map(stim_to_resp)
                stop_fail_with_resp['is_correct'] = stop_fail_with_resp['key_press'] == stop_fail_with_resp['expected_response']
                metrics[f'{paired_cond}_stop_fail_acc'] = stop_fail_with_resp['is_correct'].mean()
            else:
                metrics[f'{paired_cond}_stop_fail_acc'] = np.nan
        else:
            metrics[f'{paired_cond}_stop_fail_acc'] = np.nan
    else:
        metrics[f'{paired_cond}_stop_fail_acc'] = np.nan

    metrics[f'{paired_cond}_stop_success'] = len(df[stop_succ_mask])/len(df[stop_mask]) if len(df[stop_mask]) > 0 else np.nan
    
    # Calculate SSRT for this condition
    metrics[f'{paired_cond}_ssrt'] = compute_SSRT(df, condition_mask=paired_mask, stim_cols=stim_cols)
    
    return metrics

def parse_dual_task_condition(paired_cond, paired_task_col):
    """
    Parse dual task condition to create appropriate mask.
    
    Args:
        paired_cond (str): Condition name to parse
        paired_task_col (str): Column name for paired task
        
    Returns:
        tuple: (mask_function, args) for creating the mask
    """
    if paired_task_col is not None:
        return lambda df: df[paired_task_col] == paired_cond, None
    else:
        # For combined conditions like n-back, parse the condition name
        if 'back' in paired_cond:
            # Parse n-back condition like "0_1back" -> n_back_condition="0", delay="1"
            parts = paired_cond.split('_')
            if len(parts) >= 2:
                n_back_condition = parts[0]
                delay = parts[1].replace('back', '')
                # Convert delay to float since df['delay'] contains floats
                delay_float = float(delay)
                return lambda df: (df['n_back_condition'] == n_back_condition) & (df['delay'] == delay_float), None
        elif paired_cond.startswith('t') and '_c' in paired_cond:
            # Parse cued task switching condition like "tstay_cstay" -> task_condition="stay", cue_condition="stay"
            task_part = paired_cond[1:paired_cond.index('_c')]  # Extract "stay" from "tstay_cstay"
            cue_part = paired_cond[paired_cond.index('_c')+2:]  # Extract "stay" from "tstay_cstay"
            return lambda df: (df['task_condition'] == task_part) & (df['cue_condition'] == cue_part), None
        else:
            return None, None

def compute_stop_signal_metrics(df, dual_task = False, paired_task_col=None, paired_conditions=None, stim_col=None, stim_cols=[]):
    """
    Compute stop signal metrics for single stop signal tasks or dual tasks with stop signal.
    - df: DataFrame
    - dual_task: if True, handle dual task
    - paired_task_col: column name for the paired task (if dual)
    - paired_conditions: list of paired task conditions (if dual)
    - stim_col: single stimulus column for mapping
    - stim_cols: multiple stimulus columns for mapping
    Returns: dict of metrics
    """
    if not dual_task:
        # Single stop signal task
        metrics = calculate_single_stop_signal_metrics(df)
        
        # Add SSD stats
        ssd_stats = calculate_stop_signal_ssd_stats(df)
        metrics.update(ssd_stats)
        
        # Add SSRT
        metrics['ssrt'] = compute_SSRT(df)
        
        return metrics
    else:
        # Dual stop signal task
        metrics = {}
        
        for paired_cond in paired_conditions:
            # Parse condition and create mask
            mask_func, args = parse_dual_task_condition(paired_cond, paired_task_col)
            if mask_func is None:
                print(f'  WARNING: Could not parse condition "{paired_cond}"')
                continue
                
            paired_mask = mask_func(df)
            
            # Calculate metrics for this condition
            condition_metrics = calculate_dual_stop_signal_condition_metrics(
                df, paired_cond, paired_mask, stim_col, stim_cols
            )
            metrics.update(condition_metrics)
        
        # Add SSD stats (calculated across all stop trials)
        ssd_stats = calculate_stop_signal_ssd_stats(df)
        metrics.update(ssd_stats)
        # Note: SSRT is now calculated per condition in calculate_dual_stop_signal_condition_metrics
        
        return metrics

def get_go_trials_rt(df, max_go_rt=2000, condition_mask=None):
    """
    Get sorted go trial reaction times with replacement for missing values.
    
    Args:
        df (pd.DataFrame): DataFrame containing task data
        max_go_rt (float): Maximum RT to use for missing values
        
    Returns:
        pd.Series: Sorted go trial RTs
    """
    if condition_mask is not None:
        go_trials = df[(df['SS_trial_type'] == 'go') & condition_mask]
    else:
        go_trials = df[df['SS_trial_type'] == 'go']
    
    go_replacement_df = go_trials.copy()
    # Replace both NaN and -1 values with max_go_rt
    go_replacement_df['rt'] = go_replacement_df['rt'].replace([np.nan, -1], max_go_rt)
    return go_replacement_df['rt'].sort_values(ascending=True, ignore_index=True)

def get_stop_trials_info(df, condition_mask=None, stim_cols=None):
    """
    Get stop trial information including failure rate and average SSD.
    
    Args:
        df (pd.DataFrame): DataFrame containing task data
        
    Returns:
        tuple: (p_respond, avg_SSD) where p_respond is probability of responding on stop trials
    """
    if condition_mask is not None:
        stop_df = df[(df['SS_trial_type'] == 'stop') & condition_mask]
    else:
        stop_df = df[df['SS_trial_type'] == 'stop']
    
    if len(stop_df) == 0:
        return 0.0, np.nan
    
    stop_failure = stop_df[stop_df['rt'] > 0]
    p_respond = len(stop_failure) / len(stop_df)
    avg_SSD = stop_df['SS_delay'].mean()
    
    return p_respond, avg_SSD

def get_nth_rt(sorted_go_rt, p_respond):
    """
    Get the nth RT from sorted go trial RTs based on stop failure probability.
    
    Args:
        sorted_go_rt (pd.Series): Sorted go trial RTs
        p_respond (float): Probability of responding on stop trials
        
    Returns:
        float: The nth RT value
    """
    if len(sorted_go_rt) == 0:
        return np.nan
    
    nth_index = int(np.rint(p_respond * len(sorted_go_rt))) - 1
    
    if nth_index < 0:
        return sorted_go_rt.iloc[0]
    elif nth_index >= len(sorted_go_rt):
        return sorted_go_rt.iloc[-1]
    else:
        return sorted_go_rt.iloc[nth_index]

def compute_SSRT(df, condition_mask=None, max_go_rt=2000, stim_cols=[]):
    """
    Compute Stop Signal Reaction Time (SSRT).
    
    Args:
        df (pd.DataFrame): DataFrame containing task data
        max_go_rt (float): Maximum RT to use for missing values
        
    Returns:
        float: SSRT value
    """
    # Get go trial RTs
    sorted_go_rt = get_go_trials_rt(df, max_go_rt, condition_mask)
    
    # Get stop trial information
    p_respond, avg_SSD = get_stop_trials_info(df, condition_mask, stim_cols)
    
    # Get nth RT
    nth_rt = get_nth_rt(sorted_go_rt, p_respond)
    # Calculate SSRT
    if avg_SSD is not None and not np.isnan(avg_SSD) and not np.isnan(nth_rt):
        return nth_rt - avg_SSD
    else:
        return np.nan
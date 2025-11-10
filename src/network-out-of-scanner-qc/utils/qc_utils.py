import pandas as pd
import os
from pathlib import Path
import re
import numpy as np

from utils.globals import (
    FLANKER_CONDITIONS,
    DIRECTED_FORGETTING_CONDITIONS,
    SPATIAL_TASK_SWITCHING_CONDITIONS,
    CUED_TASK_SWITCHING_CONDITIONS,
    SPATIAL_WITH_CUED_CONDITIONS,
    GO_NOGO_CONDITIONS,
    SHAPE_MATCHING_CONDITIONS,
    FLANKER_WITH_CUED_CONDITIONS,
    FLANKER_WITH_CUED_CONDITIONS_FMRI,
    GO_NOGO_WITH_CUED_CONDITIONS,
    SHAPE_MATCHING_WITH_CUED_CONDITIONS,
    CUED_TASK_SWITCHING_WITH_DIRECTED_FORGETTING_CONDITIONS,
    SHAPE_MATCHING_CONDITIONS_WITH_DIRECTED_FORGETTING,
)

def initialize_qc_csvs(tasks, output_path, include_session: bool = False):
    """
    Initialize QC CSV files for all tasks.
    
    Args:
        tasks (list): List of task names
        output_path (Path): Path to save QC CSVs
    """
    for task in tasks:
        columns = get_task_columns(task, include_session=include_session)
        df = pd.DataFrame(columns=columns)
        df.to_csv(output_path / f"{task}_qc.csv", index=False)

def extend_metric_columns(base_columns, conditions):
    """
    Extend base columns with acc and RT metrics for given conditions.
    
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

def infer_task_name_from_filename(fname: str) -> str | None:
    name = fname.lower()
    # Ignore practice files by caller
    parts = []
    if 'stop_signal' in name or 'stopsignal' in name or 'stop-signal' in name:
        parts.append('stop_signal')
    if 'go_nogo' in name or 'gonogo' in name or 'go-nogo' in name:
        parts.append('go_nogo')
    if 'shape_matching' in name or 'shapematching' in name or 'shape-matching' in name:
        parts.append('shape_matching')
    if 'directed_forgetting' in name or 'directedforgetting' in name or 'directed-forgetting' in name:
        parts.append('directed_forgetting')
    if 'spatial_task_switching' in name or 'spatialtaskswitching' in name or 'spatial-task-switching' in name:
        parts.append('spatial_task_switching')
    if 'flanker' in name:
        parts.append('flanker')
    if 'cued_task_switching' in name or 'cuedtaskswitching' in name or 'cued-task-switching' in name:
        parts.append('cued_task_switching')
    if 'n_back' in name or 'nback' in name or 'nback' in name or 'n-back' in name:
        parts.append('n_back')
    if not parts:
        return None
    if len(parts) == 1:
        return f"{parts[0]}_single_task_network"
    # Dual task: stable canonical order with 'with'
    # Prefer stop_signal first if present, else lexicographic for consistency
    if 'stop_signal' in parts:
        first = 'stop_signal'
        parts.remove('stop_signal')
        second = parts[0]
    else:
        parts = sorted(parts)
        first, second = parts[0], parts[1]
    return f"{first}_with_{second}"

def extend_go_nogo_metric_columns(base_columns, conditions):
    """
    Extend base columns with acc and RT metrics for go_nogo tasks.
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
        if cond.endswith('_nogo') or 'nogo' in cond or cond == 'nogo':
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

def create_stop_signal_dual_columns(paired_conditions, include_nogo_commission=False, include_nogo_metrics=False):
    """
    Create column names for stop signal dual tasks.
    
    Args:
        paired_conditions (list): List of paired task conditions
        include_nogo_commission (bool): Whether to include nogo_commission_rate column
        include_nogo_metrics (bool): Whether to include nogo-specific metrics for go/nogo tasks
        
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
    
    if include_nogo_metrics:
        conditions.extend([
            "nogo_go_acc",
            "nogo_stop_success_rate"
        ])
    
    return conditions

def get_task_columns(task_name, sample_df=None, include_session: bool = False):
    """
    Define columns for each task's QC CSV.
    """
    base_columns = ['subject_id', 'session'] if include_session else ['subject_id']
    
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
                return base_columns + create_stop_signal_dual_columns(go_conditions, include_nogo_commission=True, include_nogo_metrics=True)
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
                cols = base_columns + create_stop_signal_dual_columns(spatial_conditions)
                return cols
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
                cols = base_columns + create_stop_signal_dual_columns(conditions)
                return cols
            return base_columns
        elif ('stop_signal' in task_name or 'stopSignal' in task_name) and 'n_back' in task_name:
            if sample_df is not None:
                conditions = []
                for n_back_condition in sample_df['n_back_condition'].unique():
                    if pd.notna(n_back_condition):
                        conditions.append(f"{n_back_condition}_collapsed")
                        for delay in sample_df['delay'].unique():
                            if pd.notna(delay):
                                condition_name = f"{n_back_condition}_{delay}back"
                                conditions.append(condition_name)
                return base_columns + create_stop_signal_dual_columns(conditions)
            # Fallback when no sample_df is provided during initialization
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
            cols = extend_metric_columns(base_columns, conditions)
            return cols
        elif 'flanker' in task_name and 'spatial_task_switching' in task_name or 'flanker' in task_name and 'spatialTS' in task_name:
            conditions = create_dual_task_conditions(SPATIAL_TASK_SWITCHING_CONDITIONS, FLANKER_CONDITIONS)
            cols = extend_metric_columns(base_columns, conditions)
            return cols
        elif 'go_nogo' in task_name and 'spatial_task_switching' in task_name or 'go_nogo' in task_name and 'spatialTS' in task_name:
            conditions = create_dual_task_conditions(SPATIAL_TASK_SWITCHING_CONDITIONS, GO_NOGO_CONDITIONS)
            cols = extend_go_nogo_metric_columns(base_columns, conditions)
            return cols
        elif 'shape_matching' in task_name and 'spatial_task_switching' in task_name or 'shape_matching' in task_name and 'spatialTS' in task_name:
            conditions = create_dual_task_conditions(SPATIAL_TASK_SWITCHING_CONDITIONS, SHAPE_MATCHING_CONDITIONS)
            cols = extend_metric_columns(base_columns, conditions)
            return cols
        elif 'cued_task_switching' in task_name and 'spatial_task_switching' in task_name or 'CuedTS' in task_name and 'spatialTS' in task_name:
            cols = extend_metric_columns(base_columns, SPATIAL_WITH_CUED_CONDITIONS)
            return cols
        elif 'flanker' in task_name and 'cued_task_switching' in task_name or 'flanker' in task_name and 'CuedTS' in task_name:
            cols = extend_metric_columns(base_columns, FLANKER_WITH_CUED_CONDITIONS)
            return cols
        elif 'go_nogo' in task_name and 'cued_task_switching' in task_name or 'go_nogo' in task_name and 'CuedTS' in task_name:
            cols = extend_go_nogo_metric_columns(base_columns, GO_NOGO_WITH_CUED_CONDITIONS)
            return cols
        elif 'shape_matching' in task_name and 'cued_task_switching' in task_name or 'shape_matching' in task_name and 'CuedTS' in task_name:
            # Filter out conditions with 'new' in them
            filtered_conditions = [c for c in SHAPE_MATCHING_WITH_CUED_CONDITIONS if 'new' not in c]
            cols = extend_metric_columns(base_columns, filtered_conditions)
            return cols
        elif 'directed_forgetting' in task_name and 'cued_task_switching' in task_name or 'directedForgetting' in task_name and 'CuedTS' in task_name:
            cols = extend_metric_columns(base_columns, CUED_TASK_SWITCHING_WITH_DIRECTED_FORGETTING_CONDITIONS)
            return cols
        elif 'go_nogo' in task_name and 'n_back' in task_name or 'go_nogo' in task_name and 'NBack' in task_name:
            return get_dual_n_back_columns(base_columns, sample_df, 'go_nogo_condition', gonogo=True)
        elif 'flanker' in task_name and 'n_back' in task_name or 'flanker' in task_name and 'NBack' in task_name:
            return get_dual_n_back_columns(base_columns, sample_df, 'flanker_condition')
        elif 'shape_matching' in task_name and 'n_back' in task_name or 'shape_matching' in task_name and 'NBack' in task_name:
            return get_dual_n_back_columns(base_columns, sample_df, 'shape_matching_condition')
        elif 'directed_forgetting' in task_name and 'n_back' in task_name or 'directedForgetting' in task_name and 'NBack' in task_name:
            return get_dual_n_back_columns(base_columns, sample_df, 'directed_forgetting_condition')
        elif 'n_back' in task_name and 'cued_task_switching' in task_name or 'NBack' in task_name and 'CuedTS' in task_name:
            cols = get_dual_n_back_columns(base_columns, sample_df, cuedts=True)
            return cols
        elif 'n_back' in task_name and 'spatial_task_switching' in task_name or 'NBack' in task_name and 'spatialTS' in task_name:
            cols = get_dual_n_back_columns(base_columns, sample_df, 'task_switch')
            return cols
    else:
        if 'spatial_task_switching' in task_name or 'spatialTS' in task_name:
            cols = extend_metric_columns(base_columns, SPATIAL_TASK_SWITCHING_CONDITIONS)
            return cols
        elif 'cued_task_switching' in task_name or 'cuedTS' in task_name:
            cols = extend_metric_columns(base_columns, CUED_TASK_SWITCHING_CONDITIONS)
            return cols
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
    Check if the task is a dual task by counting distinct task components.
    
    A dual task contains 2 or more distinct task components. Task components are:
    - stop_signal
    - go_nogo
    - n_back
    - flanker
    - cued_task_switching
    - spatial_task_switching
    - shape_matching
    - directed_forgetting
    
    This works for both predefined names (e.g., "flanker_with_cued_task_switching") 
    and inferred names (e.g., "cued_task_switching_with_flanker").
    """
    # Define task components (order matters - check longer names first to avoid partial matches)
    task_components = [
        'cued_task_switching',
        'spatial_task_switching',
        'directed_forgetting',
        'shape_matching',
        'stop_signal',
        'go_nogo',
        'n_back',
        'flanker'
    ]
    
    # Count how many distinct task components appear in the task name
    count = 0
    for component in task_components:
        if component in task_name:
            count += 1
    
    return count >= 2

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
    if task_name == 'cued_task_switching_with_flanker':
        # For in-scanner flanker+cued: include both test_trial and test_cue rows
        # test_cue contains the stay/switch information needed for condition matching
        if 'trial_id' in df.columns:
            filtered = df[df['trial_id'].isin(['test_trial', 'test_cue'])]
            return filtered if len(filtered) > 0 else df
        return df
    
    if 'trial_id' in df.columns:
        filtered = df[df['trial_id'] == 'test_trial']
        # If filter removes everything (in-scanner may not label), fall back to original
        return filtered if len(filtered) > 0 else df
    return df

def preprocess_rt_tail_cutoff(df: pd.DataFrame, subject_id: str | None = None, session: str | None = None, task_name: str | None = None, last_n_test_trials: int = 10):
    """
    Detects if the experiment was terminated early by finding the last valid
    response ('rt' != -1) within 'test_trial' rows. If any trials exist after
    this last valid response, they are considered a "tail" and are trimmed.

    The cutoff removes ALL rows (including fixations, etc.) after the
    last valid test trial.

    Returns a tuple: (df_trimmed, cutoff_index_within_test_trials, cutoff_before_halfway, proportion_blank)

    If no tail is found to trim, returns (df, None, False, proportion_blank).
    """
    if 'trial_id' not in df.columns or 'rt' not in df.columns:
        # Still calculate proportion of blank trials
        if 'rt' in df.columns:
            df['rt'] = pd.to_numeric(df['rt'], errors='coerce').fillna(-1)
            if len(df) > 0:
                proportion_blank = (df['rt'] == -1).sum() / len(df)
            else:
                proportion_blank = 0.0
        else:
            proportion_blank = 0.0
        return df, None, False, proportion_blank

    # Ensure 'rt' column is numeric; treat NaN as non-response (-1)
    df['rt'] = pd.to_numeric(df['rt'], errors='coerce').fillna(-1)

    # Find the last row (across ALL rows) with a valid response
    valid_mask_all = df['rt'] != -1
    if not valid_mask_all.any():
        # No valid responses at all; nothing to trim specially here
        # Calculate proportion of blank trials
        if 'trial_id' in df.columns:
            test_trials = df[df['trial_id'] == 'test_trial']
            if len(test_trials) > 0:
                proportion_blank = 1.0  # All are blank
            else:
                proportion_blank = 0.0
        else:
            proportion_blank = 1.0  # All are blank
        return df, None, False, proportion_blank

    last_valid_idx = valid_mask_all[valid_mask_all].index[-1]
    if last_valid_idx == df.index[-1]:
        # Already ends with a valid response; no trailing -1 segment
        # Still calculate proportion of blank trials
        if 'trial_id' in df.columns:
            test_trials = df[df['trial_id'] == 'test_trial']
            if len(test_trials) > 0:
                blank_trials = (test_trials['rt'] == -1).sum()
                proportion_blank = blank_trials / len(test_trials)
            else:
                proportion_blank = 0.0
        else:
            if len(df) > 0:
                blank_trials = (df['rt'] == -1).sum()
                proportion_blank = blank_trials / len(df)
            else:
                proportion_blank = 0.0
        return df, None, False, proportion_blank

    # Verify that ALL rows after last_valid_idx are indeed -1
    tail_all_minus1 = (df.loc[last_valid_idx:].iloc[1:]['rt'] == -1).all()
    if not tail_all_minus1:
        # Mixed tail; do not trim
        # Still calculate proportion of blank trials
        if 'trial_id' in df.columns:
            test_trials = df[df['trial_id'] == 'test_trial']
            if len(test_trials) > 0:
                blank_trials = (test_trials['rt'] == -1).sum()
                proportion_blank = blank_trials / len(test_trials)
            else:
                proportion_blank = 0.0
        else:
            if len(df) > 0:
                blank_trials = (df['rt'] == -1).sum()
                proportion_blank = blank_trials / len(df)
            else:
                proportion_blank = 0.0
        return df, None, False, proportion_blank

    # Additional guard: require that the last last_n_test_trials test_trial RTs are -1
    if 'trial_id' in df.columns:
        df_test_end = df[df['trial_id'] == 'test_trial']
        if len(df_test_end) < last_n_test_trials:
            # Not enough trailing test trials to be confident; do not trim
            # Still calculate proportion of blank trials
            if len(df_test_end) > 0:
                blank_trials = (df_test_end['rt'] == -1).sum()
                proportion_blank = blank_trials / len(df_test_end)
            else:
                proportion_blank = 0.0
            return df, None, False, proportion_blank
        if not (pd.to_numeric(df_test_end['rt'].tail(last_n_test_trials), errors='coerce').fillna(-1) == -1).all():
            # Do not trim unless the final last_n_test_trials test trials are all -1
            # Still calculate proportion of blank trials
            if len(df_test_end) > 0:
                blank_trials = (df_test_end['rt'] == -1).sum()
                proportion_blank = blank_trials / len(df_test_end)
            else:
                proportion_blank = 0.0
            return df, None, False, proportion_blank

    # Trim to include up to and including last_valid_idx
    cutoff_iloc = df.index.get_loc(last_valid_idx)
    df_trimmed = df.iloc[:cutoff_iloc + 1].copy()

    # Compute cutoff position relative to ALL rows
    cutoff_pos = cutoff_iloc + 1  # first dropped row position in full df (0-based index within df)
    halfway = len(df) / 2.0
    cutoff_before_halfway = cutoff_pos < halfway
    
    # Calculate proportion of blank trials (rt == -1) in original dataframe
    if 'trial_id' in df.columns:
        test_trials = df[df['trial_id'] == 'test_trial']
        if len(test_trials) > 0:
            blank_trials = (test_trials['rt'] == -1).sum()
            proportion_blank = blank_trials / len(test_trials)
        else:
            proportion_blank = 0.0
    else:
        # If no trial_id, check all rows
        if len(df) > 0:
            blank_trials = (df['rt'] == -1).sum()
            proportion_blank = blank_trials / len(df)
        else:
            proportion_blank = 0.0

    return df_trimmed, cutoff_pos, cutoff_before_halfway, proportion_blank

def sort_subject_ids(df):
    # Only process rows where subject_id is a string starting with 's'
    df = df.copy()
    df['subject_id_numeric'] = df['subject_id'].apply(
        lambda x: int(x.replace('s', '')) if isinstance(x, str) and x.startswith('s') else float('inf')
    )
    
    # If session column exists, extract numeric session value for sorting
    if 'session' in df.columns:
        def extract_session_numeric(x):
            if pd.isna(x) or x is None:
                return float('inf')
            # Handle formats like "ses-1", "ses-11", "1", "11", etc.
            x_str = str(x).strip()
            if x_str.startswith('ses-'):
                try:
                    return int(x_str.replace('ses-', ''))
                except ValueError:
                    return float('inf')
            else:
                try:
                    return int(x_str)
                except ValueError:
                    return float('inf')
        
        df['session_numeric'] = df['session'].apply(extract_session_numeric)
        df = df.sort_values(by=['subject_id_numeric', 'session_numeric'], ascending=True)
        df = df.drop(columns=['subject_id_numeric', 'session_numeric'])
    else:
        df = df.sort_values(by='subject_id_numeric', ascending=True)
        df = df.drop(columns=['subject_id_numeric'])
    
    return df

def update_qc_csv(output_path, task_name, subject_id, metrics, session=None):
    qc_file = output_path / f"{task_name}_qc.csv"
    try:
        df = pd.read_csv(qc_file)
        # Ensure session column exists if session is provided
        if session is not None and 'session' not in df.columns:
            df.insert(1, 'session', pd.Series(dtype=str))
        # Add any new columns from metrics that aren't in the DataFrame
        for key in metrics.keys():
            if key not in df.columns:
                df[key] = np.nan
        new_row_dict = {'subject_id': [subject_id], **metrics}
        if session is not None:
            new_row_dict['session'] = [session]
        new_row = pd.DataFrame(new_row_dict)
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
        df = sort_subject_ids(df)
        df.to_csv(qc_file, index=False)
    except FileNotFoundError:
        print(f"Warning: QC file {qc_file} not found")

def calculate_acc(df, mask_acc):
    """
    Calculate acc for given mask.
    
    Args:
        df (pd.DataFrame): DataFrame containing task data
        mask_acc (pd.Series): Boolean mask for acc calculation
        
    Returns:
        float: acc (mean of correct_trial)
    """
    correct_col = 'correct_trial' if 'correct_trial' in df.columns else 'correct'
    return df[mask_acc][correct_col].mean() if len(df[mask_acc]) > 0 else np.nan

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

def add_category_accuracies(df, column_name, label_to_metric_key, metrics, stopsignal=False, cuedts=False, gonogo=False):
    """
    Add acc metrics aggregated over all trials for specified category labels.

    Args:
        df (pd.DataFrame): Input dataframe
        column_name (str): Column to derive category labels from
        label_to_metric_key (dict): Mapping from label (lowercased) to output metric key
        metrics (dict): Metrics dict to populate
        stopsignal (bool): Whether this is a stop signal task
        cuedts (bool): Whether this is a cued task switching task
        gonogo (bool): Whether this is a go_nogo task
    """
    if column_name not in df.columns:
        return
    series = df[column_name].apply(lambda x: str(x).lower())
    for label, metric_key in label_to_metric_key.items():
        mask = series == label
        if stopsignal:
            mask = mask & (df['SS_trial_type'] == 'go') #only calculating acc for go trials
        if gonogo:
            mask = mask & (df['go_nogo_condition'] == 'go')
        if cuedts:
            metrics[metric_key] = (df[mask]['key_press'] == df[mask]['correct_response']).mean()
        else:
            metrics[metric_key] = calculate_acc(df, mask)

def calculate_basic_metrics(df, mask_acc, cond_name, metrics_dict, cued_with_flanker=False):
    """
    Calculate all basic metrics (acc, RT, omission rate, commission rate) for a condition.
    
    Args:
        df (pd.DataFrame): DataFrame containing task data
        mask_acc (pd.Series): Boolean mask for accuracy calculation
        cond_name (str): Condition name for metric keys
        metrics_dict (dict): Dictionary to store metrics
        
    Returns:
        None: Updates metrics_dict in place
    """
    # Use 'correct' if instructed and present; else default to 'correct_trial' then 'correct'
    correct_col = 'correct' if cued_with_flanker else 'correct_trial'
    
    if cued_with_flanker:
        # For cued+flanker: task_condition and cue_condition are on row N, but correct is on row N+1
        # We need to map each condition row index to the correct value from the next row position
        idx_list = list(df.index)
        correct_series = pd.Series(index=df.index, dtype=float)
        
        # Map each index to the next row's correct value
        for i, current_idx in enumerate(idx_list):
            if i + 1 < len(idx_list):
                next_idx = idx_list[i + 1]
                correct_series.loc[current_idx] = df[correct_col].loc[next_idx]
            else:
                correct_series.loc[current_idx] = np.nan
        
        # Create masks using the shifted correct values
        correct_mask = correct_series == 1
        mask_rt = mask_acc & correct_mask
        # For accuracy, use the shifted correct values for the masked rows
        acc_value = correct_series[mask_acc].mean() if mask_acc.sum() > 0 else np.nan
    else:
        correct_mask = df[correct_col] == 1
        mask_rt = mask_acc & correct_mask
        # Use calculate_acc function which properly handles the correct column
        acc_value = calculate_acc(df, mask_acc)
    
    mask_omission = mask_acc & (df['key_press'] == -1) if 'key_press' in df.columns else pd.Series([False] * len(df))
    # Commission: responded but incorrect (correct_mask == False means incorrect)
    if cued_with_flanker:
        mask_commission = mask_acc & (df['key_press'] != -1) & (~correct_mask) if 'key_press' in df.columns else pd.Series([False] * len(df))
    else:
        mask_commission = mask_acc & (df['key_press'] != -1) & (~correct_mask) if 'key_press' in df.columns else pd.Series([False] * len(df))
    
    total_num_trials = len(df[mask_acc])
    metrics_dict[f'{cond_name}_acc'] = acc_value
    metrics_dict[f'{cond_name}_rt'] = calculate_rt(df, mask_rt)
    metrics_dict[f'{cond_name}_omission_rate'] = calculate_omission_rate(df, mask_omission, total_num_trials)
    metrics_dict[f'{cond_name}_commission_rate'] = calculate_commission_rate(df, mask_commission, total_num_trials)

def calculate_go_nogo_metrics(df, mask_acc, cond_name, metrics_dict, response_equality: bool = False):
    """
    Calculate go_nogo metrics with special handling for nogo condition.
    For nogo: only calculate RT for commission errors, no omission/commission rates.
    For go: calculate all metrics normally.
    
    Args:
        df (pd.DataFrame): DataFrame containing task data
        mask_acc (pd.Series): Boolean mask for acc calculation
        cond_name (str): Condition name for metric keys
        metrics_dict (dict): Dictionary to store metrics
        
    Returns:
        None: Updates metrics_dict in place
    """
    # Check if this is a nogo condition
    is_nogo = cond_name.endswith('_nogo') or 'nogo' in cond_name or cond_name == 'nogo'
    
    if is_nogo:
        # For nogo: only calculate RT for commission errors (incorrect responses)
        mask_rt = mask_acc & (df['key_press'] != -1)
        # For in-scanner nogo, use response equality like go conditions
        if response_equality and {'correct_response','key_press'}.issubset(df.columns):
            # Coerce both to numeric for comparison, handling NaN
            kp = pd.to_numeric(df['key_press'], errors='coerce')
            cr = pd.to_numeric(df['correct_response'], errors='coerce')
            eq_series = (kp == cr).astype(int)
            metrics_dict[f'{cond_name}_acc'] = eq_series[mask_acc].mean() if len(df[mask_acc]) > 0 else np.nan
        else:
            metrics_dict[f'{cond_name}_acc'] = calculate_acc(df, mask_acc)
        metrics_dict[f'{cond_name}_rt'] = calculate_rt(df, mask_rt)
        # Don't calculate omission_rate or commission_rate for nogo
    else:
        # For go: calculate all metrics
        if response_equality and {'correct_response','key_press'}.issubset(df.columns):
            # Coerce both to numeric for comparison, handling NaN
            kp = pd.to_numeric(df['key_press'], errors='coerce')
            cr = pd.to_numeric(df['correct_response'], errors='coerce')
            eq_series = (kp == cr).astype(int)
            mask_rt = mask_acc & (eq_series == 1)
            mask_omission = mask_acc & (df['key_press'] == -1)
            # Commission: responded but incorrect (eq_series == 0 means incorrect)
            mask_commission = mask_acc & (df['key_press'] != -1) & (eq_series == 0) & eq_series.notna()
            metrics_dict[f'{cond_name}_acc'] = eq_series[mask_acc].mean() if len(df[mask_acc]) > 0 else np.nan
        else:
            correct_col = 'correct_trial' if 'correct_trial' in df.columns else 'correct'
            mask_rt = mask_acc & (df[correct_col] == 1)
            mask_omission = mask_acc & (df['key_press'] == -1)
            mask_commission = mask_acc & (df['key_press'] != -1) & (df[correct_col] == 0)
            metrics_dict[f'{cond_name}_acc'] = calculate_acc(df, mask_acc)
        total_num_trials = len(df[mask_acc])
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
                    (df['task_condition'].apply(lambda x: str(x).lower()) == task) &
                    (df['cue_condition'].apply(lambda x: str(x).lower()) == cue)
                )
                calculate_basic_metrics(df, mask_acc, cond, metrics, cued_with_flanker=True)
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
                    (df['task_condition'].apply(lambda x: str(x).lower()) == task) &
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
    if condition_type == 'directed_forgetting':
        add_category_accuracies(
            df,
            'cued_dimension',
            {'remember': 'remember_acc', 'forget': 'forget_acc'},
            metrics
        )
    elif condition_type == 'shape_matching':
        add_category_accuracies(
            df,
            'task',
            {'same': 'same_acc', 'different': 'different_acc'},
            metrics
        )
    elif condition_type == 'go_nogo':
        add_category_accuracies(
            df,
            'task',
            {'parity': 'parity_acc', 'magnitude': 'magnitude_acc'},
            metrics,
            gonogo=True
        )
    else:
        add_category_accuracies(
            df,
            'task',
            {'parity': 'parity_acc', 'magnitude': 'magnitude_acc'},
            metrics
        )
    return metrics

def compute_n_back_metrics(df, condition_list, paired_task_col=None, paired_conditions=None, cuedts=False, gonogo=False, shapematching=False, spatialts=False):
    """
    Compute metrics for n-back tasks (single, dual, or n-back with cuedts).
    - df: DataFrame
    - condition_list: list of n-back conditions (e.g., ['0', '2']) or list of tuples for duals
    - paired_task_col: column name for the paired task (if dual)
    - paired_conditions: list of paired task conditions (if dual)
    - cuedts: if True, handle n-back with cued task switching
    - gonogo: if True, handle n-back with go_nogo task switching
    - shapematching: if True, handle n-back with shape matching task switching
    - spatialts: if True, handle n-back with spatial task switching
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
        add_category_accuracies(
                df,
                'curr_task',
                {'one_back': 'one_back_acc', 'two_back': 'two_back_acc'},
                metrics
            )
        return metrics
    elif gonogo:
        for n_back_condition in df['n_back_condition'].str.lower().unique():
            if pd.isna(n_back_condition):
                continue
            for delay in df['delay'].unique():
                if paired_conditions is not None:
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
                if paired_conditions is not None:
                    for paired_cond in paired_conditions:
                        if shapematching:
                            condition = f"n_back_{n_back_condition}_{delay}back_shape_matching_{paired_cond.lower()}"
                        else:
                            condition = f"{n_back_condition}_{delay}back_{paired_cond.lower()}"
                        # For in-scanner nback+spatialTS, use task_switch_condition when present
                        effective_col = paired_task_col
                        if paired_task_col == 'task_switch' and 'task_switch_condition' in df.columns:
                            effective_col = 'task_switch_condition'
                        mask_acc = (df['n_back_condition'].str.lower() == n_back_condition) & (df['delay'] == delay) & (df[effective_col].astype(str).str.lower() == paired_cond.lower())
                        calculate_basic_metrics(df, mask_acc, condition, metrics)
        if spatialts:
            add_category_accuracies(
                df,
                'predictable_dimension',
                {'1-back': '1_back_acc', '2-back': '2_back_acc'},
                metrics
            )
    return metrics

def compute_fmri_cued_spatial_task_switching_metrics(df, condition_list):
    """
    Compute metrics for cued task switching with spatial task switching dual task (fMRI mode).
    Uses the task_switch column directly which contains combined condition values like
    'cuedtstaycstay_spatialtstaycstay'.
    
    Args:
        df (pd.DataFrame): DataFrame containing task data with 'task_switch' column
        condition_list (list): List of combined conditions (e.g., SPATIAL_WITH_CUED_CONDITIONS)
        
    Returns:
        dict: Metrics for cued + spatial task switching
    """
    metrics = {}
    if 'task_switch' not in df.columns:
        print("Warning: 'task_switch' column not found for fMRI cued+spatial task")
        return metrics
    
    for cond in condition_list:
        # Match rows where task_switch column exactly equals the condition
        mask_acc = df['task_switch'].astype(str).str.strip() == cond
        calculate_basic_metrics(df, mask_acc, cond, metrics)
    
    return metrics

def compute_out_of_scanner_cued_spatial_task_switching_metrics(df, condition_list):
    """
    Compute metrics for cued task switching with spatial task switching dual task (out-of-scanner mode).
    Parses separate task_condition, cue_condition, and task_switch columns.
    
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
    add_category_accuracies(
            df,
            'predictable_dimension',
            {'parity': 'parity_acc', 'magnitude': 'magnitude_acc'},
            metrics
        )
    return metrics

def add_overall_accuracy(metrics, df=None, task_name=None):
    """
    Add overall mean accuracy to metrics dict using calculate_acc on all test trials.
    
    Args:
        metrics (dict): Metrics dictionary
        df (pd.DataFrame, optional): DataFrame containing task data (already filtered to test trials)
        task_name (str, optional): Name of the task
        
    Returns:
        dict: Updated metrics dictionary with overall_acc
    """
    # Skip overall accuracy for go_nogo and stop_signal tasks
    if task_name and ('go_nogo' in task_name or 'stop_signal' in task_name or 'stopSignal' in task_name):
        return metrics
    
    # If dataframe is provided, calculate overall accuracy using calculate_acc on all test trials
    if df is not None:
        # Filter to test trials if not already done
        if task_name:
            df_filtered = filter_to_test_trials(df, task_name)
        else:
            df_filtered = df
        
        # Create mask for all test trials
        if len(df_filtered) > 0:
            mask_all = pd.Series([True] * len(df_filtered), index=df_filtered.index)
            if mask_all.sum() > 0:
                metrics['overall_acc'] = calculate_acc(df_filtered, mask_all)
            else:
                metrics['overall_acc'] = np.nan
        else:
            metrics['overall_acc'] = np.nan
    
    return metrics

def get_task_metrics(df, task_name, config):
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
            metrics = calculate_metrics(df, conditions, condition_columns, is_dual_task(task_name))
            return add_overall_accuracy(metrics, df, task_name)
        
        elif ('directed_forgetting' in task_name and 'go_nogo' in task_name) or ('directedForgetting' in task_name and 'go_nogo' in task_name):
            conditions = {
                'directed_forgetting': DIRECTED_FORGETTING_CONDITIONS,
                'go_nogo': GO_NOGO_CONDITIONS
            }
            condition_columns = {
                'directed_forgetting': 'directed_forgetting_condition',
                'go_nogo': 'go_nogo_condition'
            }
            metrics = calculate_metrics(df, conditions, condition_columns, is_dual_task(task_name))
            return add_overall_accuracy(metrics, df, task_name)
        
        elif ('flanker' in task_name and 'go_nogo' in task_name) or ('flanker' in task_name and 'go_nogo' in task_name):
            conditions = {
                'flanker': FLANKER_CONDITIONS,
                'go_nogo': GO_NOGO_CONDITIONS
            }
            condition_columns = {
                'flanker': 'flanker_condition',
                'go_nogo': 'go_nogo_condition'
            }
            metrics = calculate_metrics(df, conditions, condition_columns, is_dual_task(task_name))
            return add_overall_accuracy(metrics, df, task_name)
        
        elif ('directed_forgetting' in task_name and 'shape_matching' in task_name) or ('directedForgetting' in task_name and 'shape_matching' in task_name):
            conditions = {
                'directed_forgetting': DIRECTED_FORGETTING_CONDITIONS,
                'shape_matching': SHAPE_MATCHING_CONDITIONS_WITH_DIRECTED_FORGETTING
            }
            condition_columns = {
                'directed_forgetting': 'directed_forgetting_condition',
                'shape_matching': 'shape_matching_condition'
            }
            metrics = calculate_metrics(df, conditions, condition_columns, is_dual_task(task_name))
            return add_overall_accuracy(metrics, df, task_name)
        
        elif ('go_nogo' in task_name and 'shape_matching' in task_name) or ('go_nogo' in task_name and 'shape_matching' in task_name):
            conditions = {
                'go_nogo': GO_NOGO_CONDITIONS,
                'shape_matching': SHAPE_MATCHING_CONDITIONS
            }
            condition_columns = {
                'go_nogo': 'go_nogo_condition',
                'shape_matching': 'shape_matching_condition'
            }
            metrics = calculate_metrics(df, conditions, condition_columns, is_dual_task(task_name))
            return add_overall_accuracy(metrics, df, task_name)
        
        elif ('flanker' in task_name and 'shape_matching' in task_name) or ('flanker' in task_name and 'shape_matching' in task_name):
            conditions = {
                'flanker': FLANKER_CONDITIONS,
                'shape_matching': SHAPE_MATCHING_CONDITIONS
            }
            condition_columns = {
                'flanker': 'flanker_condition',
                'shape_matching': 'shape_matching_condition'
            }
            metrics = calculate_metrics(df, conditions, condition_columns, is_dual_task(task_name))
            return add_overall_accuracy(metrics, df, task_name)
        
        elif ('spatial_task_switching' in task_name and 'directed_forgetting' in task_name) or ('spatialTS' in task_name and 'directedForgetting' in task_name):
            conditions = {
                'spatial_task_switching': SPATIAL_TASK_SWITCHING_CONDITIONS,
                'directed_forgetting': DIRECTED_FORGETTING_CONDITIONS
            }
            condition_columns = {
                'spatial_task_switching': 'task_switch',
                'directed_forgetting': 'directed_forgetting_condition'
            }
            metrics = calculate_metrics(df, conditions, condition_columns, is_dual_task(task_name), spatialts=True, directedforgetting=True)
            return add_overall_accuracy(metrics, df, task_name)
        
        elif ('spatial_task_switching' in task_name and 'flanker' in task_name) or ('spatialTS' in task_name and 'flanker' in task_name):
            conditions = {
                'spatial_task_switching': SPATIAL_TASK_SWITCHING_CONDITIONS,
                'flanker': FLANKER_CONDITIONS
            }
            condition_columns = {
                'spatial_task_switching': 'task_switch',
                'flanker': 'flanker_condition'
            }
            return calculate_metrics(df, conditions, condition_columns, is_dual_task(task_name), spatialts=True)
        
        elif ('spatial_task_switching' in task_name and 'go_nogo' in task_name) or ('spatialTS' in task_name and 'go_nogo' in task_name):
            conditions = {
                'spatial_task_switching': SPATIAL_TASK_SWITCHING_CONDITIONS,
                'go_nogo': GO_NOGO_CONDITIONS
            }
            condition_columns = {
                'spatial_task_switching': 'task_switch',
                'go_nogo': 'go_nogo_condition'
            }
            metrics = calculate_metrics(df, conditions, condition_columns, is_dual_task(task_name), spatialts=True, gonogo=True)
            return add_overall_accuracy(metrics, df, task_name)
        
        elif ('spatial_task_switching' in task_name and 'shape_matching' in task_name) or ('spatialTS' in task_name and 'shape_matching' in task_name):
            conditions = {
                'spatial_task_switching': SPATIAL_TASK_SWITCHING_CONDITIONS,
                'shape_matching': SHAPE_MATCHING_CONDITIONS
            }
            condition_columns = {
                'spatial_task_switching': 'task_switch',
                'shape_matching': 'shape_matching_condition'
            }
            metrics = calculate_metrics(df, conditions, condition_columns, is_dual_task(task_name), spatialts=True, shapematching=True)
            return add_overall_accuracy(metrics, df, task_name)
        
        elif ('cued_task_switching' in task_name and 'spatial_task_switching' in task_name) or ('CuedTS' in task_name and 'spatialTS' in task_name):
            # Determine mode based on column structure
            if not config.is_fmri:
                metrics = compute_out_of_scanner_cued_spatial_task_switching_metrics(df, SPATIAL_WITH_CUED_CONDITIONS)
            else:
                metrics = compute_fmri_cued_spatial_task_switching_metrics(df, SPATIAL_WITH_CUED_CONDITIONS)
            return add_overall_accuracy(metrics, df, task_name)
        elif ('flanker' in task_name and 'cued_task_switching' in task_name) or ('flanker' in task_name and 'CuedTS' in task_name):
            if config.is_fmri:
                metrics = compute_cued_task_switching_metrics(df, FLANKER_WITH_CUED_CONDITIONS_FMRI, 'flanker', flanker_col='flanker_condition')
            else:
                metrics = compute_cued_task_switching_metrics(df, FLANKER_WITH_CUED_CONDITIONS, 'flanker', flanker_col='flanker_condition')
            return add_overall_accuracy(metrics, df, task_name)
        elif ('go_nogo' in task_name and 'cued_task_switching' in task_name) or ('go_nogo' in task_name and 'CuedTS' in task_name):
            metrics = compute_cued_task_switching_metrics(df, GO_NOGO_WITH_CUED_CONDITIONS, 'go_nogo', go_nogo_col='go_nogo_condition')
            return add_overall_accuracy(metrics, df, task_name)
        elif ('shape_matching' in task_name and 'cued_task_switching' in task_name) or ('shape_matching' in task_name and 'CuedTS' in task_name):
            # Filter out conditions with 'new' in them
            filtered_conditions = [c for c in SHAPE_MATCHING_WITH_CUED_CONDITIONS if 'new' not in c]
            metrics = compute_cued_task_switching_metrics(df, filtered_conditions, 'shape_matching', shape_matching_col='shape_matching_condition')
            # Also filter the returned metrics dictionary to remove any columns with 'new' (safety check)
            metrics = {k: v for k, v in metrics.items() if 'new' not in k}
            return add_overall_accuracy(metrics, df, task_name)
        elif ('directed_forgetting' in task_name and 'cued_task_switching' in task_name) or ('directedForgetting' in task_name and 'CuedTS' in task_name):
            metrics = compute_cued_task_switching_metrics(df, CUED_TASK_SWITCHING_WITH_DIRECTED_FORGETTING_CONDITIONS, 'directed_forgetting', directed_forgetting_col='directed_forgetting_condition')
            return add_overall_accuracy(metrics, df, task_name)
        elif ('n_back' in task_name and 'go_nogo' in task_name) or ('NBack' in task_name and 'go_nogo' in task_name):
            # Example: dual n-back with go_nogo
            paired_conditions = [c for c in df['go_nogo_condition'].unique() if pd.notna(c)]
            metrics = compute_n_back_metrics(df, None, paired_task_col='go_nogo_condition', paired_conditions=paired_conditions, gonogo=True)
            return add_overall_accuracy(metrics, df, task_name)
        elif ('n_back' in task_name and 'flanker' in task_name) or ('NBack' in task_name and 'flanker' in task_name):
            paired_conditions = [c for c in df['flanker_condition'].unique() if pd.notna(c)]
            metrics = compute_n_back_metrics(df, None, paired_task_col='flanker_condition', paired_conditions=paired_conditions)
            return add_overall_accuracy(metrics, df, task_name)
        elif ('n_back' in task_name and 'shape_matching' in task_name) or ('NBack' in task_name and 'shape_matching' in task_name):
            paired_conditions = [c for c in df['shape_matching_condition'].unique() if pd.notna(c)]
            metrics = compute_n_back_metrics(df, None, paired_task_col='shape_matching_condition', paired_conditions=paired_conditions, shapematching=True)
            return add_overall_accuracy(metrics, df, task_name)
        elif ('n_back' in task_name and 'directed_forgetting' in task_name) or ('NBack' in task_name and 'directed_forgetting' in task_name):
            paired_conditions = [c for c in df['directed_forgetting_condition'].unique() if pd.notna(c)]
            metrics = compute_n_back_metrics(df, None, paired_task_col='directed_forgetting_condition', paired_conditions=paired_conditions)
            return add_overall_accuracy(metrics, df, task_name)
        elif ('n_back' in task_name and 'cued_task_switching' in task_name) or ('NBack' in task_name and 'CuedTS' in task_name):
            metrics = compute_n_back_metrics(df, None, paired_task_col='task_switch', paired_conditions=None, cuedts=True)
            return add_overall_accuracy(metrics, df, task_name)
        elif ('n_back' in task_name and 'spatial_task_switching' in task_name) or ('NBack' in task_name and 'spatialTS' in task_name):
            spatial_col = 'task_switch_condition' if 'task_switch_condition' in df.columns else 'task_switch'
            paired_conditions = [c for c in df[spatial_col].unique() if pd.notna(c) and c != 'na']
            metrics = compute_n_back_metrics(df, None, paired_task_col=spatial_col, paired_conditions=paired_conditions, spatialts=True)
            return add_overall_accuracy(metrics, df, task_name)
        elif ('stop_signal' in task_name and 'flanker' in task_name) or ('stopSignal' in task_name and 'flanker' in task_name):
            paired_conditions = [c for c in df['flanker_condition'].unique() if pd.notna(c)]
            metrics = compute_stop_signal_metrics(df, dual_task=True, paired_task_col='flanker_condition', paired_conditions=paired_conditions, stim_col='center_letter')
            return add_overall_accuracy(metrics, df, task_name)
        elif ('stop_signal' in task_name and 'go_nogo' in task_name) or ('stopSignal' in task_name and 'go_nogo' in task_name):
            # Only process 'go' condition, not 'nogo'
            paired_conditions = ['go']  # Only process go condition
            metrics = compute_stop_signal_metrics(df, dual_task=True, paired_task_col='go_nogo_condition', paired_conditions=paired_conditions, stim_col='stim')
            
            # Calculate nogo commission rate separately
            nogo_mask = (df['go_nogo_condition'] == 'nogo')
            nogo_commission_mask = nogo_mask & (df['key_press'] != -1) & (df['correct_trial'] == 0)
            num_nogo_commissions = len(df[nogo_commission_mask])
            total_nogo_trials = len(df[nogo_mask])
            metrics['nogo_commission_rate'] = num_nogo_commissions / total_nogo_trials if total_nogo_trials > 0 else np.nan
            
            # Calculate nogo go acc (acc on go trials when nogo condition is present)
            nogo_go_mask = (df['go_nogo_condition'] == 'nogo') & (df['SS_trial_type'] == 'go')
            if len(df[nogo_go_mask]) > 0:
                nogo_go_correct = (df[nogo_go_mask]['key_press'] == df[nogo_go_mask]['correct_response']).sum()
                metrics['nogo_go_acc'] = nogo_go_correct / len(df[nogo_go_mask])
            else:
                metrics['nogo_go_acc'] = np.nan
            
            # Calculate nogo stop success rate (across all nogo stop trials)
            nogo_stop_mask = (df['go_nogo_condition'] == 'nogo') & (df['SS_trial_type'] == 'stop')
            if len(df[nogo_stop_mask]) > 0:
                nogo_stop_success = (df[nogo_stop_mask]['key_press'] == -1).astype(int)
                metrics['nogo_stop_success_rate'] = nogo_stop_success.mean()
            else:
                metrics['nogo_stop_success_rate'] = np.nan
            
            return add_overall_accuracy(metrics, df, task_name)
        elif ('stop_signal' in task_name and 'shape_matching' in task_name) or ('stopSignal' in task_name and 'shape_matching' in task_name):
            paired_conditions = [c for c in df['shape_matching_condition'].unique() if pd.notna(c)]
            metrics = compute_stop_signal_metrics(df, dual_task=True, paired_task_col='shape_matching_condition', paired_conditions=paired_conditions, stim_col='shape_matching_condition')
            return add_overall_accuracy(metrics, df, task_name)
        elif ('stop_signal' in task_name and 'directed_forgetting' in task_name) or ('stopSignal' in task_name and 'directedForgetting' in task_name):
            paired_conditions = [c for c in df['directed_forgetting_condition'].unique() if pd.notna(c)]
            metrics = compute_stop_signal_metrics(df, dual_task=True, paired_task_col='directed_forgetting_condition', paired_conditions=paired_conditions, stim_col='directed_forgetting_condition')
            return add_overall_accuracy(metrics, df, task_name)
        elif ('stop_signal' in task_name and 'spatial_task_switching' in task_name) or ('stopSignal' in task_name and 'spatialTS' in task_name):
            paired_conditions = [c for c in df['task_switch'].unique() if pd.notna(c) and c != 'na']
            metrics = compute_stop_signal_metrics(df, dual_task=True, paired_task_col='task_switch', paired_conditions=paired_conditions, stim_cols=['number', 'predictable_dimension'], spatialts=True)
            return add_overall_accuracy(metrics, df, task_name)
        elif ('stop_signal' in task_name and 'n_back' in task_name) or ('stopSignal' in task_name and 'NBack' in task_name):
            paired_conditions = []
            for n_back_condition in df['n_back_condition'].unique():
                if pd.notna(n_back_condition):
                    paired_conditions.append(f"{n_back_condition}_collapsed")
                    for delay in df['delay'].unique():
                        if pd.notna(delay):
                            paired_conditions.append(f"{n_back_condition}_{delay}back")
            metrics = compute_stop_signal_metrics(df, dual_task=True, paired_task_col=None, paired_conditions=paired_conditions, stim_col='n_back_condition')
            return add_overall_accuracy(metrics, df, task_name)
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
            metrics = compute_stop_signal_metrics(df, dual_task=True, paired_task_col=None, paired_conditions=paired_conditions, stim_cols=['stim_number', 'task'], cuedts=True)
            return add_overall_accuracy(metrics, df, task_name)
        # Add more dual n-back pairings as needed
    else:
        # Special handling for n-back task
        if 'n_back' in task_name:
            metrics = compute_n_back_metrics(df, None)
            return add_overall_accuracy(metrics, df, task_name)

        elif 'cued_task_switching' in task_name:
            metrics = compute_cued_task_switching_metrics(df, CUED_TASK_SWITCHING_CONDITIONS, 'single')
            return add_overall_accuracy(metrics, df, task_name)
        elif 'spatial_task_switching' in task_name or 'spatialTS' in task_name:
            conditions = {'spatial_task_switching': SPATIAL_TASK_SWITCHING_CONDITIONS}
            # Prefer in-scanner column when present
            spatial_col = 'task_switch_condition' if 'task_switch_condition' in df.columns else 'task_switch'
            condition_columns = {'spatial_task_switching': spatial_col}
            metrics = calculate_metrics(df, conditions, condition_columns, is_dual_task(task_name), spatialts=True)
            return add_overall_accuracy(metrics, df, task_name)
        # Special handling for stop signal task
        elif 'stop_signal' in task_name:
            metrics = compute_stop_signal_metrics(df, dual_task=False)
            return add_overall_accuracy(metrics, df, task_name)
        # For other single tasks, we only need one set of conditions
        elif 'directed_forgetting' in task_name or 'directedForgetting' in task_name:
            conditions = {'directed_forgetting': DIRECTED_FORGETTING_CONDITIONS}
            condition_columns = {'directed_forgetting': 'directed_forgetting_condition'}
        elif 'flanker' in task_name:
            conditions = {'flanker': FLANKER_CONDITIONS}
            condition_columns = {'flanker': 'flanker_condition'}
        elif 'go_nogo' in task_name:
            conditions = {'go_nogo': GO_NOGO_CONDITIONS}
            condition_columns = {'go_nogo': 'go_nogo_condition'}
        elif 'shape_matching' in task_name:
            conditions = {'shape_matching': SHAPE_MATCHING_CONDITIONS}
            condition_columns = {'shape_matching': 'shape_matching_condition'}
        else:
            print(f"Unknown task: {task_name}")
            return None
    
        metrics = calculate_metrics(df, conditions, condition_columns, is_dual_task(task_name))
        return add_overall_accuracy(metrics)

def calculate_metrics(df, conditions, condition_columns, is_dual_task, spatialts=False, shapematching=False, directedforgetting=False, gonogo=False):
    """
    Calculate RT and acc metrics for any task.
    
    Args:
        df (pd.DataFrame): DataFrame containing task data
        conditions (dict): Dictionary of task names and their conditions
        condition_columns (dict): Dictionary of task names and their condition column names
        is_dual_task (bool): Whether this is a dual task
        spatialts (bool): Whether this is a spatial task switching task
        shapematching (bool): Whether this is a shape matching task
        directedforgetting (bool): Whether this is a directed forgetting task
        gonogo (bool): Whether this is a go_nogo task
    Returns:
        dict: Dictionary containing task-specific metrics
    """
    metrics = {}
    
    if is_dual_task:
        # For dual tasks, iterate through all combinations of conditions
        task1, task2 = list(conditions.keys())
        for cond1 in conditions[task1]:
            for cond2 in conditions[task2]:
                if 'go_nogo' in task1:
                    mask1 = df[condition_columns[task1]].astype(str).str.lower() == cond1.lower()
                    mask2 = df[condition_columns[task2]].str.contains(cond2, case=False, na=False)
                    mask_acc = mask1 & mask2
                elif 'go_nogo' in task2:
                    mask1 = df[condition_columns[task1]].str.contains(cond1, case=False, na=False)
                    mask2 = df[condition_columns[task2]].astype(str).str.lower() == cond2.lower()
                    mask_acc = mask1 & mask2
                else:
                    mask_acc = df[condition_columns[task1]].str.contains(cond1, case=False, na=False) & \
                        df[condition_columns[task2]].str.contains(cond2, case=False, na=False)

                # Check if this is a go_nogo task
                if 'go_nogo' in task1 or 'go_nogo' in task2:
                    calculate_go_nogo_metrics(df, mask_acc, f'{cond1}_{cond2}', metrics)
                else:
                    calculate_basic_metrics(df, mask_acc, f'{cond1}_{cond2}', metrics)
        if spatialts and shapematching:
            add_category_accuracies(
                df,
                'predictable_dimension',
                {'the same': 'same_acc', 'different': 'different_acc'},
                metrics
            )
        elif spatialts and directedforgetting:
            add_category_accuracies(
                df,
                'predictable_dimension',
                {'remember': 'remember_acc', 'forget': 'forget_acc'},
                metrics
            )
        elif spatialts:
            add_category_accuracies(
                df,
                'predictable_dimension',
                {'parity': 'parity_acc', 'magnitude': 'magnitude_acc'},
                metrics
            )
        elif spatialts and gonogo:
            add_category_accuracies(
                df,
                'predictable_dimension',
                {'parity': 'parity_acc', 'magnitude': 'magnitude_acc'},
                metrics,
                gonogo=True
            )
    else:
        # For single tasks, just iterate through conditions
        task = list(conditions.keys())[0]
        for cond in conditions[task]:
            mask_acc = (df[condition_columns[task]] == cond)
            # Check if this is a go_nogo task
            if 'go_nogo' in task:
                # For single go_nogo: use response equality only in fMRI mode
                is_fmri = os.environ.get('QC_DATA_MODE', 'out_of_scanner').lower() == 'fmri'
                calculate_go_nogo_metrics(df, mask_acc, cond, metrics, response_equality=is_fmri)
            else:
                calculate_basic_metrics(df, mask_acc, cond, metrics)
        if spatialts:
            add_category_accuracies(
                df,
                'predictable_dimension',
                {'parity': 'parity_acc', 'magnitude': 'magnitude_acc'},
                metrics
            )

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

def normalize_flanker_conditions(df):
    """
    Normalize flanker condition values by removing h_ and f_ prefixes.
    
    Args:
        df (pd.DataFrame): DataFrame that may contain flanker_condition column
        
    Returns:
        pd.DataFrame: DataFrame with normalized flanker conditions
    """
    if 'flanker_condition' in df.columns:
        df = df.copy()
        # Map h_incongruent, h_congruent, f_incongruent, f_congruent to incongruent, congruent
        flanker_mapping = {
            'H_incongruent': 'incongruent',
            'H_congruent': 'congruent', 
            'F_incongruent': 'incongruent',
            'F_congruent': 'congruent'
        }
        df['flanker_condition'] = df['flanker_condition'].replace(flanker_mapping)
    return df

def correct_columns(csv_path):
    df = pd.read_csv(csv_path)
    columns_renamed = False
    for col in df.columns:
        if 'tswitch_new_cswitch' in col:
            new_col = col.replace('tswitch_new_cswitch', 'tswitch_cswitch')
            df.rename(columns={col: new_col}, inplace=True)
            columns_renamed = True
    
    if columns_renamed:
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
    
    # Stop failure acc based on stimulus-response mapping from go trials
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

def calculate_dual_stop_signal_condition_metrics(df, paired_cond, paired_mask, stim_col=None, stim_cols=None, cuedts=False, spatialts=False):
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
    
    # Stop failure acc based on stimulus-response mapping from go trials
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

    if cuedts:
        add_category_accuracies(
            df,
            'task',
            {'parity': 'parity_acc', 'magnitude': 'magnitude_acc'},
            metrics,
            stopsignal=True,
            cuedts=True
        )
    elif spatialts:
        add_category_accuracies(
            df,
            'predictable_dimension',
            {'parity': 'parity_acc', 'magnitude': 'magnitude_acc'},
            metrics,
            stopsignal=True
        )
    
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
        if 'collapsed' in paired_cond:
            # Handle collapsed conditions like "match_collapsed" or "mismatch_collapsed"
            # These collapse across all delays for a given n_back_condition
            n_back_condition = paired_cond.replace('_collapsed', '')
            return lambda df: (df['n_back_condition'] == n_back_condition), None
        elif 'back' in paired_cond:
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

def compute_stop_signal_metrics(df, dual_task = False, paired_task_col=None, paired_conditions=None, stim_col=None, stim_cols=[], cuedts=False, spatialts=False):
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
        
        if paired_conditions is not None:
            for paired_cond in paired_conditions:
                # Parse condition and create mask
                mask_func, args = parse_dual_task_condition(paired_cond, paired_task_col)
                if mask_func is None:
                    print(f'  WARNING: Could not parse condition "{paired_cond}"')
                    continue
                    
                paired_mask = mask_func(df)
                
                # Calculate metrics for this condition
                condition_metrics = calculate_dual_stop_signal_condition_metrics(
                    df, paired_cond, paired_mask, stim_col, stim_cols, cuedts, spatialts
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
    return go_replacement_df['rt'].sort_values(ascending=True)

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
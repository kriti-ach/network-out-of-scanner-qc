import pandas as pd
import glob
from pathlib import Path
import re
import os
import sys

from utils.qc_utils import (
    initialize_qc_csvs,
    extract_task_name_out_of_scanner,
    update_qc_csv,
    get_task_metrics,
    append_summary_rows_to_csv,
    correct_columns,
    normalize_flanker_conditions,
    preprocess_rt_tail_cutoff,
    infer_task_name_from_filename,
)
from utils.violations_utils import compute_violations, aggregate_violations, plot_violations, create_violations_matrices
from utils.globals import SINGLE_TASKS, DUAL_TASKS, LAST_N_TEST_TRIALS
from utils.exclusion_utils import check_exclusion_criteria, remove_some_flags_for_exclusion
from utils.config import load_config

# Optional CLI override: --mode=fmri or --mode=out_of_scanner
for arg in sys.argv[1:]:
    if arg.startswith('--mode='):
        os.environ['QC_DATA_MODE'] = arg.split('=', 1)[1]

cfg = load_config()
input_root = cfg.input_folder
output_path = cfg.qc_output_folder
flags_output_path = cfg.flags_output_folder
exclusions_output_path = cfg.exclusions_output_folder
violations_output_path = cfg.violations_output_folder
trimmed_records = []
last_n_test_trials = LAST_N_TEST_TRIALS

if cfg.is_fmri:
    # Discover tasks from filenames first (exclude practice)
    discovered_tasks = set()
    for subj_dir in glob.glob(str(input_root / 's*')):
        for ses_dir in glob.glob(str(Path(subj_dir) / 'ses-*')):
            for file in glob.glob(str(Path(ses_dir) / '*.csv')):
                if '/practice/' in file.lower():
                    continue
                tname = infer_task_name_from_filename(Path(file).name)
                if tname:
                    discovered_tasks.add(tname)
    tasks = sorted(discovered_tasks)
else:
    tasks = (SINGLE_TASKS + DUAL_TASKS)

# Initialize QC CSVs for all tasks (include session column for fmri mode)
initialize_qc_csvs(tasks, output_path, include_session=cfg.is_fmri)

violations_df = pd.DataFrame()
trimmed_data = []
if cfg.is_fmri:
    # In-scanner (CSV per session) iterate and process, ignoring practice
    for subj_dir in glob.glob(str(input_root / 's*')):
        subject_id = Path(subj_dir).name
        if not re.match(r"s\d{2,}", subject_id):
            continue
        print(f"Processing Subject: {subject_id}")
        for ses_dir in glob.glob(str(Path(subj_dir) / 'ses-*')):
            for file in glob.glob(str(Path(ses_dir) / '*.csv')):
                if '/practice/' in file.lower():
                    continue
                filename = Path(file).name
                task_name = infer_task_name_from_filename(filename)
                if not task_name:
                    continue
                try:
                    df = pd.read_csv(file)
                    if 'flanker' in task_name and 'stop_signal' in task_name:
                        df = normalize_flanker_conditions(df)
                    # Generic RT tail cutoff
                    df_trimmed, cut_pos, cut_before_halfway = preprocess_rt_tail_cutoff(
                        df,
                        subject_id=subject_id,
                        session=Path(ses_dir).name,
                        task_name=task_name,
                        last_n_test_trials=last_n_test_trials,
                    )
                    if cut_pos is not None:
                        trimmed_records.append({
                            'subject_id': subject_id,
                            'session': Path(ses_dir).name,
                            'task_name': task_name,
                            'cutoff_index': int(cut_pos),
                            'before_halfway': bool(cut_before_halfway),
                        })
                        if cut_before_halfway:
                            continue
                        else:
                            df = df_trimmed
                    metrics = get_task_metrics(df, task_name, cfg)
                    if (not cfg.is_fmri) and 'stop_signal' in task_name:
                        violations_df = pd.concat([violations_df, compute_violations(subject_id, df, task_name)])
                    # Session for fmri from ses-* directory name
                    session = Path(ses_dir).name if cfg.is_fmri else None
                    update_qc_csv(output_path, task_name, subject_id, metrics, session=session)
                except Exception as e:
                    print(f"Error processing {task_name} for subject {subject_id}: {str(e)}")
else:
    # Out-of-scanner: iterate per subject and CSV files
    for subject_folder in glob.glob(str(input_root / "s*")):
        subject_id = Path(subject_folder).name
        if re.match(r"s\d{2,}", subject_id):
            print(f"Processing Subject: {subject_id}")
            for file in glob.glob(str(Path(subject_folder) / "*.csv")):
                filename = Path(file).name
                task_name = extract_task_name_out_of_scanner(filename)
                if task_name == 'stop_signal_with_go_no_go':
                    task_name = 'stop_signal_with_go_nogo'
                if task_name:
                    try:
                        df = pd.read_csv(file)
                        # Normalize flanker conditions (remove h_ and f_ prefixes)
                        if 'flanker' in task_name and 'stop_signal' in task_name:
                            df = normalize_flanker_conditions(df)
                        # Generic RT tail cutoff
                        df_trimmed, cut_pos, cut_before_halfway = preprocess_rt_tail_cutoff(
                            df,
                            subject_id=subject_id,
                            session=None,
                            task_name=task_name,
                            last_n_test_trials=last_n_test_trials,
                        )
                        if cut_pos is not None:
                            trimmed_records.append({
                                'subject_id': subject_id,
                                'session': '',
                                'task_name': task_name,
                                'cutoff_index': int(cut_pos),
                                'before_halfway': bool(cut_before_halfway),
                            })
                            if cut_before_halfway:
                                continue
                            else:
                                df = df_trimmed
                        metrics = get_task_metrics(df, task_name, cfg)
                        if (not cfg.is_fmri) and 'stop_signal' in task_name:
                            violations_df = pd.concat([violations_df, compute_violations(subject_id, df, task_name)])
                        update_qc_csv(output_path, task_name, subject_id, metrics, session=None)
                    except Exception as e:
                        print(f"Error processing {task_name} for subject {subject_id}: {str(e)}")

for task in tasks:
    exclusion_df = pd.DataFrame({'subject_id': [], 'metric': [], 'metric_value': [], 'threshold': []})
    append_summary_rows_to_csv(output_path / f"{task}_qc.csv")
    if task == 'flanker_with_cued_task_switching' or task == 'shape_matching_with_cued_task_switching':
        correct_columns(output_path / f"{task}_qc.csv")
    task_csv = pd.read_csv(output_path / f"{task}_qc.csv")
    exclusion_df = check_exclusion_criteria(task, task_csv, exclusion_df)
    
    # Create a copy for flagged data (flags that will be removed)
    flagged_df = exclusion_df.copy()
    
    # Remove some flags for exclusion data
    exclusion_df = remove_some_flags_for_exclusion(task, exclusion_df)
    
    # Flagged data contains only the flags that were removed (original - filtered)
    flagged_df = flagged_df[~flagged_df.index.isin(exclusion_df.index)]
    
    # Save both datasets
    flagged_df.to_csv(flags_output_path / f"flagged_data_{task}.csv", index=False)
    exclusion_df.to_csv(exclusions_output_path / f"excluded_data_{task}.csv", index=False)
        
if not cfg.is_fmri:
    violations_df.to_csv(violations_output_path / 'violations_data.csv', index=False)
    aggregated_violations_df = aggregate_violations(violations_df)
    aggregated_violations_df.to_csv(violations_output_path / 'aggregated_violations_data.csv', index=False)
    plot_violations(aggregated_violations_df, violations_output_path)
    create_violations_matrices(aggregated_violations_df, violations_output_path)

# Save list of trimmed CSVs
if len(trimmed_records) > 0:
    trimmed_df = pd.DataFrame(trimmed_records)
    out_csv = Path('/oak/stanford/groups/russpold/data/network_grant/behavioral_data/trimmed_fmri_behavior_tasks.csv') if cfg.is_fmri else Path('/oak/stanford/groups/russpold/data/network_grant/behavioral_data/trimmed_out_of_scanner_tasks.csv')
    trimmed_df.to_csv(out_csv, index=False)
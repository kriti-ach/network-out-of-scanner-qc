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
)
from utils.violations_utils import compute_violations, aggregate_violations, plot_violations, create_violations_matrices
from utils.globals import SINGLE_TASKS_OUT_OF_SCANNER, DUAL_TASKS_OUT_OF_SCANNER
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

def infer_task_name_from_filename(fname: str) -> str | None:
    name = fname.lower()
    # Ignore practice files by caller
    parts = []
    if 'stop_signal' in name:
        parts.append('stop_signal')
    if 'go_nogo' in name:
        parts.append('go_nogo')
    if 'flanker' in name:
        parts.append('flanker')
    if 'shape_matching' in name:
        parts.append('shape_matching')
    if 'directed_forgetting' in name:
        parts.append('directed_forgetting')
    if 'spatial_task_switching' in name:
        parts.append('spatial_task_switching')
    if 'cued_task_switching' in name or 'cuedts' in name:
        parts.append('cued_task_switching')
    if 'n_back' in name or 'nback' in name:
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
    tasks = (SINGLE_TASKS_OUT_OF_SCANNER + DUAL_TASKS_OUT_OF_SCANNER)

# Initialize QC CSVs for all tasks
initialize_qc_csvs(tasks, output_path)

violations_df = pd.DataFrame()

if cfg.is_fmri:
    # In-scanner (CSV per session) iterate and process, ignoring practice
    for subj_dir in glob.glob(str(input_root / 's*')):
        subject_id = Path(subj_dir).name
        if not re.match(r"s\d{2,}", subject_id):
            continue
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
                    metrics = get_task_metrics(df, task_name)
                    if (not cfg.is_fmri) and 'stop_signal' in task_name:
                        violations_df = pd.concat([violations_df, compute_violations(subject_id, df, task_name)])
                    update_qc_csv(output_path, task_name, subject_id, metrics)
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
                        metrics = get_task_metrics(df, task_name)
                        if (not cfg.is_fmri) and 'stop_signal' in task_name:
                            violations_df = pd.concat([violations_df, compute_violations(subject_id, df, task_name)])
                        update_qc_csv(output_path, task_name, subject_id, metrics)
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
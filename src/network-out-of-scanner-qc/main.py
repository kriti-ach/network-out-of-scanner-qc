import pandas as pd
import glob
from pathlib import Path
import re

from utils.utils import (
    initialize_qc_csvs,
    extract_task_name_out_of_scanner,
    extract_task_name_fmri,
    update_qc_csv,
    get_task_metrics,
    append_summary_rows_to_csv,
    correct_columns,
)
from utils.violations_utils import compute_violations
from utils.globals import SINGLE_TASKS_FMRI, DUAL_TASKS_FMRI, SINGLE_TASKS_OUT_OF_SCANNER, DUAL_TASKS_OUT_OF_SCANNER

# folder_path = Path("/oak/stanford/groups/russpold/data/network_grant/validation_BIDS/")
# output_path = Path("/oak/stanford/groups/russpold/data/network_grant/behavioral_data/qc_by_task/")

folder_path = Path("/oak/stanford/groups/russpold/data/network_grant/behavioral_data/out_of_scanner")
output_path = Path("/oak/stanford/groups/russpold/data/network_grant/behavioral_data/qc_by_task/out_of_scanner/")
violations_output_path = Path("/oak/stanford/groups/russpold/data/network_grant/behavioral_data/out_of_scanner_violations")

# Initialize QC CSVs for all tasks
initialize_qc_csvs(SINGLE_TASKS_OUT_OF_SCANNER + DUAL_TASKS_OUT_OF_SCANNER, output_path)

violations_list = []
for subject_folder in glob.glob(str(folder_path / "s*")):
    subject_id = Path(subject_folder).name
    # if re.match(r"s\d{2,}", subject_id):
    if re.match(r"s\d{2,}", subject_id):
        print(f"Processing Subject: {subject_id}")

        # for file in glob.glob(str(Path(subject_folder) / "*csv")):
        for file in glob.glob(str(Path(subject_folder) / "*.csv")):
            # session = Path(file).parent.parent.name
            # run = Path(file).stem.split('_')[3]
            filename = Path(file).name
            task_name = extract_task_name_out_of_scanner(filename)
            if task_name == 'stop_signal_with_go_no_go':
                task_name = 'stop_signal_with_go_nogo'
            # task_name = extract_task_name_fmri(filename)
            print(f"Processing task: {task_name}")
            
            if task_name:
                try:
                    df = pd.read_csv(file)
                    # df = pd.read_csv(file, sep='\t')
                    metrics = get_task_metrics(df, task_name)
                    if 'stop_signal' in task_name:
                        print(f'Computing violations for {task_name}')
                        violations = compute_violations(subject_id, df, task_name)
                        if not violations.empty:
                            violations_list.append(violations)
                    update_qc_csv(output_path, task_name, subject_id, metrics)
                except Exception as e:
                    print(f"Error processing {task_name} for subject {subject_id}: {str(e)}")

for task in SINGLE_TASKS_OUT_OF_SCANNER + DUAL_TASKS_OUT_OF_SCANNER:
    append_summary_rows_to_csv(output_path / f"{task}_qc.csv")
    if task == 'flanker_with_cued_task_switching' or task == 'shape_matching_with_cued_task_switching':
        correct_columns(output_path / f"{task}_qc.csv")

violations_df = pd.concat(violations_list, ignore_index=True)
violations_df.to_csv(violations_output_path / 'violations_data.csv', index=False)
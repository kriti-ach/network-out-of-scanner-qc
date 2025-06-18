import pandas as pd
import glob
from pathlib import Path
import re

from utils.utils import (
    initialize_qc_csvs,
    extract_task_name,
    update_qc_csv,
    get_task_metrics
)

from utils.globals import SINGLE_TASKS, DUAL_TASKS

folder_path = Path("/oak/stanford/groups/russpold/data/network_grant/behavioral_data/network_fmri_raw_final_2/")
output_path = Path("/oak/stanford/groups/russpold/data/network_grant/behavioral_data/qc_by_task/")

# Initialize QC CSVs for all tasks
initialize_qc_csvs(SINGLE_TASKS + DUAL_TASKS, output_path)

for subject_folder in glob.glob(str(folder_path / "*")):
    subject_id = Path(subject_folder).name
    if re.match(r"s\d{2,}", subject_id):
        print(f"Processing Subject: {subject_id}")

        for file in glob.glob(str(Path(subject_folder) / "*.csv")):
            filename = Path(file).name
            task_name = extract_task_name(filename)
            print(f"Processing task: {task_name}")
            
            if task_name:
                try:
                    df = pd.read_csv(file)
                    metrics = get_task_metrics(df, task_name)
                    update_qc_csv(output_path, task_name, subject_id, metrics)
                except Exception as e:
                    print(f"Error processing {task_name} for subject {subject_id}: {str(e)}")
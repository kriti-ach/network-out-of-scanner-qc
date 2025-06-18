import pandas as pd
import glob
from pathlib import Path
import re

from utils.utils import (
    initialize_qc_csvs,
    extract_task_name_out_of_scanner,
    extract_task_name_fmri,
    update_qc_csv,
    get_task_metrics
)

from utils.globals import SINGLE_TASKS_FMRI, DUAL_TASKS_FMRI, SINGLE_TASKS_OUT_OF_SCANNER, DUAL_TASKS_OUT_OF_SCANNER

folder_path = Path("/oak/stanford/groups/russpold/data/network_grant/validation_BIDS/")
output_path = Path("/oak/stanford/groups/russpold/data/network_grant/behavioral_data/qc_by_task/")

# Initialize QC CSVs for all tasks
initialize_qc_csvs(SINGLE_TASKS_FMRI + DUAL_TASKS_FMRI, output_path)

for subject_folder in glob.glob(str(folder_path / "sub-s*")):
    subject_id = Path(subject_folder).name
    # if re.match(r"s\d{2,}", subject_id):
    if re.match(r"sub-s\d{2,}", subject_id):
        print(f"Processing Subject: {subject_id}")

        # for file in glob.glob(str(Path(subject_folder) / "*csv")):
        for file in glob.glob(str(Path(subject_folder) / "ses-*" / "func" / "*.tsv")):
            session = Path(file).parent.parent.name
            run = Path(file).stem.split('_')[3]
            filename = Path(file).name
            # task_name = extract_task_name_out_of_scanner(filename)
            task_name = extract_task_name_fmri(filename)
            print(f"Processing task: {task_name}")
            
            if task_name:
                if task_name == 'cuedTS':
                    print(file)
                    # check if file exists
                    if not Path(file).exists():
                        print(f"File {file} does not exist")
                        continue
                try:
                    # df = pd.read_csv(file)
                    df = pd.read_csv(file, sep='\t')
                    metrics = get_task_metrics(df, task_name)
                    update_qc_csv(output_path, task_name, subject_id, session, run, metrics)
                except Exception as e:
                    print(f"Error processing {task_name} for subject {subject_id}: {str(e)}")
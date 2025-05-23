import pandas as pd
import glob
from pathlib import Path
import re

from utils.utils import (
    initialize_qc_csvs,
    extract_task_name,
    update_qc_csv
)

folder_path = Path("/oak/stanford/groups/russpold/data/network_grant/behavioral_data/out_of_scanner/") 

# Initialize QC CSVs
initialize_qc_csvs()

for subject_folder in glob.glob(str(folder_path / "*")):
    subject_id = Path(subject_folder).name
    if re.match(r"s\d{2,}", subject_id):
        print(f"Processing Subject: {subject_id}")

        for file in glob.glob(str(Path(subject_folder) / "*.csv")):
            filename = Path(file).name
            task_name = extract_task_name(filename)
            print(task_name)
            if task_name:
                df = pd.read_csv(file)
                # Add your score calculation logic here
                # score = calculate_score(df)
                # update_qc_csv(task_name, subject_id, score)
import pandas as pd
from utils.qc_utils import initialize_qc_csvs, update_qc_csv
from pathlib import Path
import tempfile

def test_initialize_and_update_qc_csvs(tmp_path: Path):
    tasks = ['stop_signal_with_flanker']
    output = tmp_path
    initialize_qc_csvs(tasks, output)
    qc_file = output / f"{tasks[0]}_qc.csv"
    # File should exist and have headers
    df = pd.read_csv(qc_file)
    assert 'subject_id' in df.columns

    # Update with a metric row
    metrics = {'congruent_go_rt': 0.5}
    update_qc_csv(output, tasks[0], 's01', metrics)
    df2 = pd.read_csv(qc_file)
    assert (df2['subject_id'] == 's01').any()
    assert 'congruent_go_rt' in df2.columns



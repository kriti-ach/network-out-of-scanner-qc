import sys
from pathlib import Path
import glob
import pandas as pd

# Ensure src package is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src" / "network-out-of-scanner-qc"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
QC_DIR = SRC_DIR / "qc"
if str(QC_DIR) not in sys.path:
    sys.path.insert(0, str(QC_DIR))

from .qc_utils import (  # noqa: E402
    initialize_qc_csvs,
    extract_task_name_out_of_scanner,
    update_qc_csv,
    get_task_metrics,
    append_summary_rows_to_csv,
    correct_columns,
)
from utils.globals import (  # noqa: E402
    SINGLE_TASKS_OUT_OF_SCANNER,
    DUAL_TASKS_OUT_OF_SCANNER,
)


def run_qc() -> None:
    folder_path = Path(
        "/oak/stanford/groups/russpold/data/network_grant/behavioral_data/out_of_scanner"
    )
    output_path = Path(
        "/oak/stanford/groups/russpold/data/network_grant/behavioral_data/out_of_scanner_qc"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    initialize_qc_csvs(SINGLE_TASKS_OUT_OF_SCANNER + DUAL_TASKS_OUT_OF_SCANNER, output_path)

    for subject_folder in glob.glob(str(folder_path / "s*")):
        subject_id = Path(subject_folder).name
        if subject_id and subject_id.startswith("s"):
            for file in glob.glob(str(Path(subject_folder) / "*.csv")):
                filename = Path(file).name
                task_name = extract_task_name_out_of_scanner(filename)
                if task_name == "stop_signal_with_go_no_go":
                    task_name = "stop_signal_with_go_nogo"
                if task_name:
                    try:
                        df = pd.read_csv(file)
                        metrics = get_task_metrics(df, task_name)
                        update_qc_csv(output_path, task_name, subject_id, metrics)
                    except Exception as e:  # pragma: no cover
                        print(f"Error processing {task_name} for subject {subject_id}: {str(e)}")

    for task in SINGLE_TASKS_OUT_OF_SCANNER + DUAL_TASKS_OUT_OF_SCANNER:
        append_summary_rows_to_csv(output_path / f"{task}_qc.csv")
        if task in ("flanker_with_cued_task_switching", "shape_matching_with_cued_task_switching"):
            correct_columns(output_path / f"{task}_qc.csv")


if __name__ == "__main__":
    run_qc()



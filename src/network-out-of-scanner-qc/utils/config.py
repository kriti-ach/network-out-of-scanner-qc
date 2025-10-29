import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PathConfig:
    input_folder: Path
    qc_output_folder: Path
    flags_output_folder: Path
    exclusions_output_folder: Path
    violations_output_folder: Path
    file_glob: str
    file_ext: str
    is_fmri: bool


def load_config() -> PathConfig:
    """Load configuration for paths based on QC_DATA_MODE env var.

    Modes:
    - out_of_scanner (default): CSV files under per-subject folders
    - fmri: BIDS events TSV files under BIDS tree
    """
    mode = os.environ.get("QC_DATA_MODE", "out_of_scanner").lower()

    if mode == "fmri":
        # In-scanner (fMRI) behavior: cleaned CSVs per session under raw_cleaned/s*/ses-*/
        input_folder = Path("/oak/stanford/groups/russpold/data/network_grant/behavioral_data/raw_cleaned")
        qc_output = Path("/oak/stanford/groups/russpold/data/network_grant/behavioral_data/qc_by_task/")
        flags_output = Path("/oak/stanford/groups/russpold/data/network_grant/behavioral_data/flags/")
        exclusions_output = Path("/oak/stanford/groups/russpold/data/network_grant/behavioral_data/exclusions/")
        violations_output = Path("/oak/stanford/groups/russpold/data/network_grant/behavioral_data/violations/")
        return PathConfig(
            input_folder=input_folder,
            qc_output_folder=qc_output,
            flags_output_folder=flags_output,
            exclusions_output_folder=exclusions_output,
            violations_output_folder=violations_output,
            file_glob="s*/ses-*/*.csv",
            file_ext=".csv",
            is_fmri=True,
        )

    # Default: out-of-scanner behavior
    input_folder = Path("/oak/stanford/groups/russpold/data/network_grant/behavioral_data/out_of_scanner")
    qc_output = Path("/oak/stanford/groups/russpold/data/network_grant/behavioral_data/out_of_scanner_qc")
    flags_output = Path("/oak/stanford/groups/russpold/data/network_grant/behavioral_data/out_of_scanner_flags")
    exclusions_output = Path("/oak/stanford/groups/russpold/data/network_grant/behavioral_data/out_of_scanner_exclusions")
    violations_output = Path("/oak/stanford/groups/russpold/data/network_grant/behavioral_data/out_of_scanner_violations")
    return PathConfig(
        input_folder=input_folder,
        qc_output_folder=qc_output,
        flags_output_folder=flags_output,
        exclusions_output_folder=exclusions_output,
        violations_output_folder=violations_output,
        file_glob="s*/**/*.csv",
        file_ext=".csv",
        is_fmri=False,
    )



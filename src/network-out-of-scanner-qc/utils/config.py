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
    # BIDS paths for scan time extraction
    discovery_bids_path: Path
    validation_bids_path: Path
    # Discovery subjects list
    discovery_subjects: list[str]
    # Trimmed CSV output path
    trimmed_csv_output_path: Path


def load_config() -> PathConfig:
    """Load configuration for paths based on QC_DATA_MODE env var.

    Modes:
    - out_of_scanner (default): CSV files under per-subject folders
    - fmri: BIDS events TSV files under BIDS tree
    """
    mode = os.environ.get("QC_DATA_MODE", "out_of_scanner").lower()

    # BIDS paths (same for both modes)
    discovery_bids_path = Path("/oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_20250402")
    validation_bids_path = Path("/oak/stanford/groups/russpold/data/network_grant/validation_BIDS")
    
    # Discovery subjects
    discovery_subjects = ['sub-s03', 'sub-s10', 'sub-s19', 'sub-s29', 'sub-s43']
    
    # Trimmed CSV output path
    trimmed_csv_output_path = Path("/oak/stanford/groups/russpold/data/network_grant/behavioral_data/behavior_cut_short")
    
    if mode == "fmri":
        # In-scanner (fMRI) behavior: cleaned CSVs per session under raw_cleaned/s*/ses-*/
        input_folder = Path("/oak/stanford/groups/russpold/data/network_grant/behavioral_data/raw_cleaned")
        qc_output = Path("/oak/stanford/groups/russpold/data/network_grant/behavioral_data/fmri_behavior_qc_by_task/")
        flags_output = Path("/oak/stanford/groups/russpold/data/network_grant/behavioral_data/fmri_behavior_flags/")
        exclusions_output = Path("/oak/stanford/groups/russpold/data/network_grant/behavioral_data/fmri_behavior_exclusions/")
        violations_output = Path("/oak/stanford/groups/russpold/data/network_grant/behavioral_data/fmri_behavior_violations/")
        return PathConfig(
            input_folder=input_folder,
            qc_output_folder=qc_output,
            flags_output_folder=flags_output,
            exclusions_output_folder=exclusions_output,
            violations_output_folder=violations_output,
            file_glob="s*/ses-*/*.csv",
            file_ext=".csv",
            is_fmri=True,
            discovery_bids_path=discovery_bids_path,
            validation_bids_path=validation_bids_path,
            discovery_subjects=discovery_subjects,
            trimmed_csv_output_path=trimmed_csv_output_path,
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
        discovery_bids_path=discovery_bids_path,
        validation_bids_path=validation_bids_path,
        discovery_subjects=discovery_subjects,
        trimmed_csv_output_path=trimmed_csv_output_path,
    )



"""
Script to process trimmed behavioral CSVs and add scan time from BIDS data.

This script:
1. Processes trimmed CSVs from discovery and validation BIDS paths
2. Changes output path to in_scanner_behavior_cut_short
3. For each trimmed CSV, finds the total scan time from BIDS data
4. Adds scan time as a column to the trimmed CSVs
"""
import pandas as pd
import json
import glob
from pathlib import Path
import re
import nibabel as nib
import numpy as np
import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent))

from utils.trimmed_behavior_utils import preprocess_rt_tail_cutoff, get_bids_task_name
from utils.config import load_config

# Load config to get paths
cfg = load_config()

# Get paths from config
DISCOVERY_BIDS_PATH = cfg.discovery_bids_path
VALIDATION_BIDS_PATH = cfg.validation_bids_path
DISCOVERY_SUBJECTS = cfg.discovery_subjects

def get_scan_time_from_bids(subject_id, session, task_name, bids_path):
    """
    Get total scan time for a subject/session from BIDS data.
    
    Looks for:
    1. JSON sidecar files with 'Duration' field
    2. NIfTI files and calculates duration from header
    
    Returns total scan time in seconds, or None if not found.
    """
    print(f"Getting scan time for {subject_id} {session} {task_name} from {bids_path}")
    subject_path = bids_path / f'sub-{subject_id}'
    if not subject_path.exists():
        return None
    print(f"Subject path exists: {subject_path}")
    
    session_path = subject_path / f'{session}'
    if not session_path.exists():
        return None
    print(f"Session path exists: {session_path}")
    total_duration = 0.0
    
    # Look for JSON sidecar files (func, beh, etc.)
    print(f"Looking for JSON files in {session_path} for task {task_name}")
    json_files = list(session_path.glob(f'**/**task-{task_name}*_echo-2.json'))
    print(f"Found {len(json_files)} JSON files in {session_path}")
    print(json_files)
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Check for Duration field
                if 'Duration' in data:
                    duration = float(data['Duration'])
                    total_duration += duration
                # Also check for RepetitionTime and number of volumes
                elif 'RepetitionTime' in data and 'NumVolumes' in data:
                    tr = float(data['RepetitionTime'])
                    n_vols = int(data['NumVolumes'])
                    duration = tr * n_vols
                    total_duration += duration
                elif 'RepetitionTime' in data:
                    # Try to get number of volumes from corresponding NIfTI file
                    nii_file = json_file.with_suffix('.nii.gz')
                    if not nii_file.exists():
                        nii_file = json_file.with_suffix('.nii')
                    if nii_file.exists():
                        try:
                            nii = nib.load(str(nii_file))
                            n_vols = nii.shape[-1] if len(nii.shape) > 3 else 1
                            tr = float(data['RepetitionTime'])
                            duration = tr * n_vols
                            total_duration += duration
                        except Exception:
                            pass
        except Exception as e:
            print(f"Warning: Could not read JSON file {json_file}: {e}")
            continue
    
    # If no duration found in JSON, try NIfTI files directly
    if total_duration == 0.0:
        nii_files = list(session_path.glob('**/*.nii.gz')) + list(session_path.glob('**/*.nii'))
        for nii_file in nii_files:
            try:
                nii = nib.load(str(nii_file))
                # Get TR from header or default to 2.0 seconds
                tr = nii.header.get_zooms()[-1] if len(nii.shape) > 3 else 2.0
                if not isinstance(tr, (int, float)) or tr <= 0:
                    tr = 2.0
                n_vols = nii.shape[-1] if len(nii.shape) > 3 else 1
                duration = tr * n_vols
                total_duration += duration
            except Exception as e:
                print(f"Warning: Could not read NIfTI file {nii_file}: {e}")
                continue
    
    return total_duration if total_duration > 0 else None


def process_trimmed_csvs():
    """
    Process behavioral CSVs listed in trimmed_fmri_behavior_tasks.csv,
    apply trimming, and add scan time from BIDS data.
    """
    # Read the trimmed tasks CSV created by main.py
    trimmed_tasks_file = cfg.trimmed_csv_output_path / 'trimmed_fmri_behavior_tasks.csv'
    if not trimmed_tasks_file.exists():
        print(f"Error: {trimmed_tasks_file} not found. Run main.py first to create it.")
        return
    
    trimmed_tasks_df = pd.read_csv(trimmed_tasks_file)
    print(f"Found {len(trimmed_tasks_df)} trimmed tasks to process")
    
    all_trimmed_data = []
    
    for idx, row in trimmed_tasks_df.iterrows():
        subject_id = row['subject_id']
        session = row['session']
        task_name = row['task_name']

        task_name = get_bids_task_name(task_name)
        
        print(f"Processing {subject_id} {session} {task_name}...")
        
        # Determine which BIDS path to use
        if subject_id in DISCOVERY_SUBJECTS:
            bids_path = DISCOVERY_BIDS_PATH
        else:
            bids_path = VALIDATION_BIDS_PATH
        
        try:
            # Get scan time from BIDS
            scan_time = get_scan_time_from_bids(subject_id, session, task_name, bids_path)
            # Add scan_time column
            trimmed_tasks_df['scan_time_seconds'] = scan_time if scan_time is not None else np.nan
            # Record metadata
            all_trimmed_data.append({
                'subject_id': subject_id,
                'session': session,
                'task_name': task_name,
                'scan_time_seconds': scan_time,
            })
            
        except Exception as e:
            print(f"  Error processing {subject_id} {session} {task_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary CSV
    if all_trimmed_data:
        summary_df = pd.DataFrame(all_trimmed_data)
        summary_file = cfg.trimmed_csv_output_path / 'trimmed_fmri_csvs_with_scan_time.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to: {summary_file}")
        print(f"Total files processed: {len(all_trimmed_data)}")
    else:
        print("\nNo files were processed.")


if __name__ == '__main__':
    process_trimmed_csvs()


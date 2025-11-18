# Network Behavior QC

## Description

This repository provides quality control (QC) tools for network behavioral data. It processes behavioral task data from both fMRI (in-scanner) and out-of-scanner sessions, generating task-specific QC reports with comprehensive metrics including accuracy, reaction time (RT), omission rate, commission rate, and condition-specific performance measures.

The pipeline automatically:
- Processes single-task and dual-task behavioral data
- Computes task-specific metrics for each subject and session
- Identifies data quality issues and exclusion criteria violations
- Generates flagged data reports and exclusion summaries
- Creates violation analyses for stop signal tasks (out-of-scanner only)
- Handles RT tail cutoff preprocessing to remove blank trials
- Supports both fMRI and out-of-scanner data modes

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kriti-ach/network-behavior-qc.git
cd network-behavior-qc
```

2. Set up the environment:
```bash
source setup_env.sh
```

This script will:
- Install `uv` package manager
- Create and activate a virtual environment
- Install all dependencies (including dev dependencies)

## Usage

### Running the QC Pipeline

The main script can be run in two modes:

**For fMRI (in-scanner) data:**
```bash
uv run src/network-behavior-qc/main.py --mode=fmri
```

**For out-of-scanner data:**
```bash
uv run src/network-behavior-qc/main.py --mode=out_of_scanner
```

### Configuration

The pipeline uses configuration settings defined in `src/network-behavior-qc/utils/config.py`. This includes:
- Input and output folder paths
- BIDS data paths (for fMRI mode)
- Task discovery and processing settings

**Note:** Paths are currently hardcoded for the Stanford Oak filesystem. Modify `config.py` to use different paths for your environment.

## Supported Tasks

### Single Tasks
- Cued Task Switching
- Directed Forgetting
- Flanker
- Go/No-Go
- N-Back
- Spatial Task Switching
- Shape Matching
- Stop Signal

### Dual Tasks
The pipeline supports all combinations of the above single tasks, including:
- Flanker with Cued Task Switching
- Stop Signal with Go/No-Go
- N-Back with various tasks
- And many more combinations (see `utils/globals.py` for complete list)

## Repository Structure

```
network-behavior-qc/
├── src/
│   └── network-behavior-qc/
│       ├── __init__.py
│       ├── main.py                    # Main QC processing script
│       ├── process_trimmed_with_scan_time.py
│       ├── trim_event_files.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── config.py              # Configuration and path settings
│       │   ├── exclusion_utils.py     # Exclusion criteria checking
│       │   ├── globals.py             # Task names, conditions, thresholds
│       │   ├── qc_utils.py            # Core QC metric computation
│       │   ├── trimmed_behavior_utils.py  # RT tail cutoff preprocessing
│       │   └── violations_utils.py    # Stop signal violation analysis
│       └── tests/                     # Unit tests
│           ├── test_basic_metrics.py
│           ├── test_csv_operations.py
│           ├── test_cued_task_switching_metrics.py
│           ├── test_exclusion_utils.py
│           ├── test_normalize_flanker_conditions.py
│           ├── test_stop_signal_metrics.py
│           └── test_violations_utils.py
│           └── test_exclusion_utils.py
│           └── test_trimmed_behavior_utils.py
├── pyproject.toml                     # Project dependencies and metadata
├── setup_env.sh                       # Environment setup script
└── README.md
```

## Output Files

The pipeline generates several types of output files:

### QC Reports
- `{task}_qc.csv`: Task-specific QC metrics for each subject/session
  - Includes accuracy, RT, omission rate, commission rate by condition
  - Summary rows with mean, median, std, and count statistics

### Flagged Data
- `flagged_data_{task}.csv`: Subjects/sessions that meet flagging criteria but may not be excluded
  - Includes condition-specific accuracy and omission rate flags (fMRI mode)

### Exclusion Data
- `excluded_data_{task}.csv`: Subjects/sessions that meet exclusion criteria
- `combined_exclusions.csv`: Aggregated exclusion data across all tasks

### Violations Analysis (Out-of-scanner only)
- `violations_data.csv`: Trial-level violation data for stop signal tasks
- `aggregated_violations_data.csv`: Subject-level violation summaries
- Violation plots and matrices (saved as image files)

### Trimmed Data Records
- `trimmed_fmri_behavior_tasks.csv` or `trimmed_out_of_scanner_tasks.csv`: Records of data trimming operations
  - Documents when RT tail cutoff was applied
  - Includes cutoff index, proportion of blank trials, and whether cutoff occurred before halfway point

## Key Features

### RT Tail Cutoff Preprocessing
The pipeline automatically detects and removes blank trials at the end of task runs using a tail cutoff algorithm. If more than half the trials are blank, the entire session is skipped.

### Exclusion Criteria
Task-specific exclusion criteria are applied based on performance thresholds:
- Accuracy thresholds (varies by task and condition)
- Omission rate thresholds
- Commission rate thresholds
- Stop signal success rate thresholds
- Go RT thresholds

See `utils/globals.py` for specific threshold values.

### Condition-Specific Metrics
Metrics are computed separately for each experimental condition (e.g., congruent/incongruent for Flanker, go/nogo for Go/No-Go tasks).

## Testing

Run the test suite using:
```bash
uv run pytest
```

Tests are located in `src/network-behavior-qc/tests/` and cover:
- Basic metric computations
- CSV operations
- Task-specific metric calculations
- Exclusion criteria checking
- Violation analysis
- Data normalization functions

## Dependencies

Core dependencies:
- `numpy>=2.2.3`
- `pandas>=2.2.3`
- `matplotlib>=3.10.0`
- `seaborn>=0.13.2`
- `nibabel>=5.0.1`

Development dependencies:
- `pytest>=8.3.4`
- `ipykernel>=6.29.5`
- `ipython>=8.32.0`
- `python-dotenv>=1.0.1`

## Notes

- This repository uses the Cookiecutter template from: https://github.com/lobennett/uv_cookie
- The pipeline requires Python >=3.12
- Paths in `config.py` are configured for Stanford Oak filesystem - modify as needed for your environment
- Practice sessions are automatically excluded from processing
- For fMRI mode, tasks are auto-discovered from filenames in the input directory structure

## Author

Kriti Achyutuni (kritiach@stanford.edu)

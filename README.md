# Network out-of-scanner QC

## Description
This repo quality controls the network out-of-scanner behavioral data. It generates task-specific QC reports with metrics like accuracy, RT, omission rate, commission rate, etc. It uses data from `/oak/stanford/groups/russpold/data/network_grant/behavioral_data/out_of_scanner` and outputs QC csvs at `/oak/stanford/groups/russpold/data/network_grant/behavioral_data/qc_by_task/out_of_scanner/`.

## Installation
Clone the repository using:

```bash
git clone https://github.com/kriti-ach/network-out-of-scanner-qc.git
```
Go into the repo using:

```bash
cd /path/to/network-out-of-scanner-qc
```

Set up the environment using:
```bash
source setup_env.sh
```
## Repository Structure

- /src:    
    - /network-out-of-scanner-qc:
        * [main.py]: Script to run the QC scripts. 
        - /utils: 
            * [utils.py]: Helper functions to condense `main.py` script.
            * [globals.py]: Contains the global variables, including the task names and dual task conditions.
        - /tests:
            - Contains tests for major functions.
            - Run tests using `uv run pytest`

## Notes

This repository used the Cookiecutter template from: https://github.com/lobennett/uv_cookie
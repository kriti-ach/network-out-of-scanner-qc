import pandas as pd
import os
from pathlib import Path
import re
import numpy as np
from utils.utils import filter_to_test_trials

def compute_violations(subject_id, df, task_name):
    violations_row = []
    delay = 1 if task_name == 'stop_signal_with_n_back' else 2

    df = filter_to_test_trials(df, task_name)

    for i in range(len(df) - delay): 
        # Check for a Go trial followed by a Stop trial with a violation
        if (df.iloc[i]['stop_signal_condition'] == 'go' and
            df.iloc[i + delay]['stop_signal_condition'] == 'stop' and
            (df.iloc[i + delay]['rt'] != -1)):
            
            go_rt = df.iloc[i]['rt']         # RT of Go trial
            stop_rt = df.iloc[i + delay]['rt']  # RT of Stop trial
            
            if pd.notna(go_rt) and pd.notna(stop_rt):  # Ensure RTs are valid
                difference = stop_rt - go_rt  # Calculate the difference
                ssd = df.iloc[i + delay]['SS_delay']    # SSD for the Stop trial
                violations_row.append({'subject_id': subject_id, 'task_name': task_name, 'ssd': ssd, 'difference': difference})

    return pd.DataFrame(violations_row)


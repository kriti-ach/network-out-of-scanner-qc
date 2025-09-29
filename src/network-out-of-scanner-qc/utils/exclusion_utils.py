import pandas as pd
import os
from pathlib import Path
import re
import numpy as np

from utils.globals import (
    STOP_SUCCESS_ACC_LOW_THRESHOLD,
    STOP_SUCCESS_ACC_HIGH_THRESHOLD,
    GO_RT_THRESHOLD,
    GO_ACC_THRESHOLD_GO_NOGO,
    NOGO_ACC_THRESHOLD_GO_NOGO,
    MISMATCH_1_BACK_THRESHOLD,
    MISMATCH_2_BACK_THRESHOLD,
    MATCH_1_BACK_THRESHOLD,
    MATCH_2_BACK_THRESHOLD,
    MISMATCH_3_BACK_THRESHOLD,
    MATCH_3_BACK_THRESHOLD,
    ACC_THRESHOLD,
    OMISSION_RATE_THRESHOLD,
    LOWER_WEIGHT,
    UPPER_WEIGHT
)
from utils.qc_utils import sort_subject_ids

def check_exclusion_criteria(task_name, task_csv, exclusion_df):
        if 'stop_signal' in task_name:
            exclusion_df = check_stop_signal_exclusion_criteria(task_name, task_csv, exclusion_df)
        if 'go_nogo' in task_name:
            exclusion_df = check_go_nogo_exclusion_criteria(task_name, task_csv, exclusion_df)
        if 'n_back' in task_name:
            exclusion_df = check_n_back_exclusion_criteria(task_name, task_csv, exclusion_df)
        return exclusion_df
        # if ('flanker' in task_name or 'directed_forgetting' in task_name or 
        # 'shape_matching' in task_name or 'spatial_task_switching' in task_name or 
        # 'cued_task_switching' in task_name):
        #     check_other_exclusion_criteria(task_name, task_csv, exclusion_df)

def compare_to_threshold(metric_name, metric_value, threshold):
    """Check if a metric value violates the exclusion criteria."""
    if 'match' in metric_name or 'mismatch' in metric_name:
        print(f'metric_name: {metric_name}, metric_value: {metric_value}, threshold: {threshold}')
        print(f'metric value < threshold: {metric_value < threshold}')
        return metric_value < threshold
    else:
        print(f'metric_name: {metric_name}, metric_value: {metric_value}, threshold: {threshold}')
        print(f'metric value > threshold: {metric_value > threshold}')
        return metric_value > threshold

def append_exclusion_row(exclusion_df, subject_id, metric_name, metric_value, threshold):
    """Append a new exclusion row to the exclusion dataframe."""
    exclusion_df = pd.concat([
        exclusion_df,
        pd.DataFrame({
            'subject_id': [subject_id],
            'metric': [metric_name],
            'metric_value': [metric_value],
            'threshold': [threshold]
        })
    ], ignore_index=True)
    return exclusion_df

def check_stop_signal_exclusion_criteria(task_name, task_csv, exclusion_df):
    for index, row in task_csv.iterrows():
        #ignore the last 4 rows (summary rows)
        if index >= len(task_csv) - 4:
            continue
        subject_id = row['subject_id']

        # Get actual column names for each metric type
        stop_success_cols = [col for col in task_csv.columns if 'stop_success' in col]
        go_rt_cols = [col for col in task_csv.columns if 'go_rt' in col]
        go_acc_cols = [col for col in task_csv.columns if 'go_acc' in col]
        go_omission_rate_cols = [col for col in task_csv.columns if 'go_omission_rate' in col]
        
        # Check stop_success specifically for low and high thresholds
        for col_name in stop_success_cols:
            value = row[col_name]
            if compare_to_threshold('stop_success_low', value, STOP_SUCCESS_ACC_LOW_THRESHOLD):
                exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, STOP_SUCCESS_ACC_LOW_THRESHOLD)
            if compare_to_threshold('stop_success_high', value, STOP_SUCCESS_ACC_HIGH_THRESHOLD):
                exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, STOP_SUCCESS_ACC_HIGH_THRESHOLD)

        # Check go_rt columns
        for col_name in go_rt_cols:
            value = row[col_name]
            if compare_to_threshold('go_rt', value, GO_RT_THRESHOLD):
                exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, GO_RT_THRESHOLD)

        # Check go_acc columns
        for col_name in go_acc_cols:
            value = row[col_name]
            if compare_to_threshold('go_acc', value, ACC_THRESHOLD):
                exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, ACC_THRESHOLD)

        # Check go_omission_rate columns
        for col_name in go_omission_rate_cols:
            value = row[col_name]
            if compare_to_threshold('go_omission_rate', value, OMISSION_RATE_THRESHOLD):
                exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, OMISSION_RATE_THRESHOLD)

    #sort by subject_id
    exclusion_df = sort_subject_ids(exclusion_df)
    return exclusion_df

def check_go_nogo_exclusion_criteria(task_name, task_csv, exclusion_df):
    for index, row in task_csv.iterrows():
        #ignore the last 4 rows (summary rows)
        if index >= len(task_csv) - 4:
            continue
        subject_id = row['subject_id']

        # Get actual column names for each metric type
        go_acc_cols = [col for col in task_csv.columns if 'go' in col and 'acc' in col and 'nogo' not in col]
        nogo_acc_cols = [col for col in task_csv.columns if 'nogo' in col and 'acc' in col]
        go_omission_rate_cols = [col for col in task_csv.columns if 'go' in col and 'omission_rate' in col and 'nogo' not in col]

        # If go accuracy < threshold AND nogo accuracy < threshold, then exclude
        # Only check when the prefix (before go_acc/nogo_acc) matches
        for col_name_go in go_acc_cols:
            for col_name_nogo in nogo_acc_cols:
                # Extract prefix before 'go_acc' and 'nogo_acc'
                go_prefix = col_name_go.replace('go_acc', '')
                nogo_prefix = col_name_nogo.replace('nogo_acc', '')
                
                # Only proceed if prefixes match
                if go_prefix == nogo_prefix:
                    go_acc_value = row[col_name_go]
                    nogo_acc_value = row[col_name_nogo]
                    if compare_to_threshold('go_acc', go_acc_value, GO_ACC_THRESHOLD_GO_NOGO) and compare_to_threshold('_nogo_acc', nogo_acc_value, NOGO_ACC_THRESHOLD_GO_NOGO):
                        exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name_go, go_acc_value, GO_ACC_THRESHOLD_GO_NOGO)
                        exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name_nogo, nogo_acc_value, NOGO_ACC_THRESHOLD_GO_NOGO)
                    if np.mean([go_acc_value, nogo_acc_value]) < ACC_THRESHOLD:
                        exclusion_df = append_exclusion_row(exclusion_df, subject_id, 'mean_go_and_nogo_acc', np.mean([go_acc_value, nogo_acc_value]), ACC_THRESHOLD)

        for col_name in go_omission_rate_cols:
            value = row[col_name]
            if compare_to_threshold('go_omission_rate', value, OMISSION_RATE_THRESHOLD):
                exclusion_df = append_exclusion_row(exclusion_df, subject_id, col_name, value, OMISSION_RATE_THRESHOLD)
    #sort by subject_id
    if len(exclusion_df) != 0:
        exclusion_df = sort_subject_ids(exclusion_df)
    return exclusion_df

def check_n_back_exclusion_criteria(task_name, task_csv, exclusion_df):
    for index, row in task_csv.iterrows():
        #ignore the last 4 rows (summary rows)
        if index >= len(task_csv) - 4:
            continue
        subject_id = row['subject_id']
        
        # Pull needed columns safely
        match_1_acc = [col for col in task_csv.columns if 'match_1.0back' in col and 'acc' in col and 'mismatch' not in col]
        mismatch_1_acc = [col for col in task_csv.columns if 'mismatch_1.0back' in col and 'acc' in col]
        match_2_acc = [col for col in task_csv.columns if 'match_2.0back' in col and 'acc' in col and 'mismatch' not in col]
        mismatch_2_acc = [col for col in task_csv.columns if 'mismatch_2.0back' in col and 'acc' in col]
        match_3_acc = [col for col in task_csv.columns if 'match_3.0back' in col and 'acc' in col and 'mismatch' not in col]
        mismatch_3_acc = [col for col in task_csv.columns if 'mismatch_3.0back' in col and 'acc' in col]

        match_1_omiss = [col for col in task_csv.columns if 'match_1.0back' in col and 'omission_rate' in col and 'mismatch' not in col]
        match_2_omiss = [col for col in task_csv.columns if 'match_2.0back' in col and 'omission_rate' in col and 'mismatch' not in col]
        mismatch_1_omiss = [col for col in task_csv.columns if 'mismatch_1.0back' in col and 'omission_rate' in col]
        mismatch_2_omiss = [col for col in task_csv.columns if 'mismatch_2.0back' in col and 'omission_rate' in col]
        match_3_omiss = [col for col in task_csv.columns if 'match_3.0back' in col and 'omission_rate' in col and 'mismatch' not in col]
        mismatch_3_omiss = [col for col in task_csv.columns if 'mismatch_3.0back' in col and 'omission_rate' in col]

        # 1-back weighted mean (20% match, 80% mismatch) < .55
        for col_name_match_1 in match_1_acc:
            for col_name_mismatch_1 in mismatch_1_acc:
                match_1_prefix = col_name_match_1.replace('match_1.0back', '')
                mismatch_1_prefix = col_name_mismatch_1.replace('mismatch_1.0back', '')
                if match_1_prefix == mismatch_1_prefix:
                    match_1_acc_value = row[col_name_match_1]
                    mismatch_1_acc_value = row[col_name_mismatch_1]
                    if not pd.isna(match_1_acc_value) and not pd.isna(mismatch_1_acc_value):
                        weighted_1 = LOWER_WEIGHT * match_1_acc_value + UPPER_WEIGHT * mismatch_1_acc_value
                        if compare_to_threshold('weighted_mean_1.0back_acc', weighted_1, ACC_THRESHOLD):
                            exclusion_df = append_exclusion_row(
                                exclusion_df, subject_id, 'weighted_mean_1.0back_acc', weighted_1, ACC_THRESHOLD
                            )  

        # 2-back weighted mean (20% match, 80% mismatch) < .55
        for col_name_match_2 in match_2_acc:
            for col_name_mismatch_2 in mismatch_2_acc:
                match_2_prefix = col_name_match_2.replace('match_2.0back', '')
                mismatch_2_prefix = col_name_mismatch_2.replace('mismatch_2.0back', '')
                if match_2_prefix == mismatch_2_prefix:
                    match_2_acc_value = row[col_name_match_2]
                    mismatch_2_acc_value = row[col_name_mismatch_2]
                    if not pd.isna(match_2_acc_value) and not pd.isna(mismatch_2_acc_value):
                        weighted_2 = LOWER_WEIGHT * match_2_acc_value + UPPER_WEIGHT * mismatch_2_acc_value
                        if compare_to_threshold('weighted_mean_2.0back_acc', weighted_2, ACC_THRESHOLD):
                            exclusion_df = append_exclusion_row(
                                exclusion_df, subject_id, 'weighted_mean_2.0back_acc', weighted_2, ACC_THRESHOLD
                            )
        # 3-back weighted mean (20% match, 80% mismatch) < .55
        for col_name_match_3 in match_3_acc:
            for col_name_mismatch_3 in mismatch_3_acc:
                match_3_prefix = col_name_match_3.replace('match_3.0back', '')
                mismatch_3_prefix = col_name_mismatch_3.replace('mismatch_3.0back', '')
                if match_3_prefix == mismatch_3_prefix:
                    match_3_acc_value = row[col_name_match_3]
                    mismatch_3_acc_value = row[col_name_mismatch_3]
                    if not pd.isna(match_3_acc_value) and not pd.isna(mismatch_3_acc_value):
                        weighted_3 = LOWER_WEIGHT * match_3_acc_value + UPPER_WEIGHT * mismatch_3_acc_value
                        if compare_to_threshold('weighted_mean_3.0back_acc', weighted_3, ACC_THRESHOLD):
                            exclusion_df = append_exclusion_row(
                                exclusion_df, subject_id, 'weighted_mean_3.0back_acc', weighted_3, ACC_THRESHOLD
                            )
        # mismatch_1 < .8 AND match_1 < .2
        for col_name_mismatch_1 in mismatch_1_acc:
            for col_name_match_1 in match_1_acc:
                mismatch_1_prefix = col_name_mismatch_1.replace('mismatch_1.0back', '')
                match_1_prefix = col_name_match_1.replace('match_1.0back', '')
                if mismatch_1_prefix == match_1_prefix:
                    mismatch_1_acc_value = row[col_name_mismatch_1]
                    match_1_acc_value = row[col_name_match_1]
                    if not pd.isna(mismatch_1_acc_value) and not pd.isna(match_1_acc_value):
                        if [compare_to_threshold('mismatch_1.0back_acc', mismatch_1_acc_value, MISMATCH_1_BACK_THRESHOLD) 
                        and compare_to_threshold('match_1.0back_acc', match_1_acc_value, MATCH_1_BACK_THRESHOLD)]:
                            exclusion_df = append_exclusion_row(
                                exclusion_df, subject_id, 'mismatch_1.0back_acc', mismatch_1_acc_value, MISMATCH_1_BACK_THRESHOLD
                            )
                            exclusion_df = append_exclusion_row(
                                exclusion_df, subject_id, 'match_1.0back_acc', match_1_acc_value, MATCH_1_BACK_THRESHOLD
                            )

        # mismatch_2 < .8 AND match_2 < .2
        for col_name_mismatch_2 in mismatch_2_acc:
            for col_name_match_2 in match_2_acc:
                mismatch_2_prefix = col_name_mismatch_2.replace('mismatch_2.0back', '')
                match_2_prefix = col_name_match_2.replace('match_2.0back', '')
                if mismatch_2_prefix == match_2_prefix:
                    mismatch_2_acc_value = row[col_name_mismatch_2]
                    match_2_acc_value = row[col_name_match_2]
                    if not pd.isna(mismatch_2_acc_value) and not pd.isna(match_2_acc_value):
                        if [compare_to_threshold('mismatch_2.0back_acc', mismatch_2_acc_value, MISMATCH_2_BACK_THRESHOLD) 
                        and compare_to_threshold('match_2.0back_acc', match_2_acc_value, MATCH_2_BACK_THRESHOLD)]:
                            exclusion_df = append_exclusion_row(
                                exclusion_df, subject_id, 'mismatch_2.0back_acc', mismatch_2_acc_value, MISMATCH_2_BACK_THRESHOLD
                            )
                            exclusion_df = append_exclusion_row(
                                exclusion_df, subject_id, 'match_2.0back_acc', match_2_acc_value, MATCH_2_BACK_THRESHOLD
                            )


        # mismatch_3 < .8 AND match_3 < .2
        for col_name_mismatch_3 in mismatch_3_acc:
            for col_name_match_3 in match_3_acc:
                mismatch_3_prefix = col_name_mismatch_3.replace('mismatch_3.0back', '')
                match_3_prefix = col_name_match_3.replace('match_3.0back', '')
                if mismatch_3_prefix == match_3_prefix:
                    mismatch_3_acc_value = row[col_name_mismatch_3]
                    match_3_acc_value = row[col_name_match_3]
                    if not pd.isna(mismatch_3_acc_value) and not pd.isna(match_3_acc_value):
                        if [compare_to_threshold('mismatch_3.0back_acc', mismatch_3_acc_value, MISMATCH_3_BACK_THRESHOLD) 
                        and compare_to_threshold('match_3.0back_acc', match_3_acc_value, MATCH_3_BACK_THRESHOLD)]:
                            exclusion_df = append_exclusion_row(
                                exclusion_df, subject_id, 'mismatch_3.0back_acc', mismatch_3_acc_value, MISMATCH_3_BACK_THRESHOLD
                            )
                            exclusion_df = append_exclusion_row(
                                exclusion_df, subject_id, 'match_3.0back_acc', match_3_acc_value, MATCH_3_BACK_THRESHOLD
                            )

        # Omission rate checks (> .5)
        for col_name_match_1 in match_1_omiss:
            value = row[col_name_match_1]
            if compare_to_threshold('match_1.0back_omission_rate', value, OMISSION_RATE_THRESHOLD):
                exclusion_df = append_exclusion_row(
                    exclusion_df, subject_id, 'match_1.0back_omission_rate', value, OMISSION_RATE_THRESHOLD
                )
        for col_name_match_2 in match_2_omiss:
            value = row[col_name_match_2]
            if compare_to_threshold('match_2.0back_omission_rate', value, OMISSION_RATE_THRESHOLD):
                exclusion_df = append_exclusion_row(
                    exclusion_df, subject_id, 'match_2.0back_omission_rate', value, OMISSION_RATE_THRESHOLD
                )
        for col_name_match_3 in match_3_omiss:
            value = row[col_name_match_3]
            if compare_to_threshold('match_3.0back_omission_rate', value, OMISSION_RATE_THRESHOLD):
                exclusion_df = append_exclusion_row(
                    exclusion_df, subject_id, 'match_3.0back_omission_rate', value, OMISSION_RATE_THRESHOLD
                )
        for col_name_mismatch_1 in mismatch_1_omiss:
            value = row[col_name_mismatch_1]
            if compare_to_threshold('mismatch_1.0back_omission_rate', value, OMISSION_RATE_THRESHOLD):
                exclusion_df = append_exclusion_row(
                    exclusion_df, subject_id, 'mismatch_1.0back_omission_rate', value, OMISSION_RATE_THRESHOLD
                )
        for col_name_mismatch_2 in mismatch_2_omiss:
            value = row[col_name_mismatch_2]
            if compare_to_threshold('mismatch_2.0back_omission_rate', value, OMISSION_RATE_THRESHOLD):
                exclusion_df = append_exclusion_row(
                    exclusion_df, subject_id, 'mismatch_2.0back_omission_rate', value, OMISSION_RATE_THRESHOLD
                )
        for col_name_mismatch_3 in mismatch_3_omiss:
            value = row[col_name_mismatch_3]
            if compare_to_threshold('mismatch_3.0back_omission_rate', value, OMISSION_RATE_THRESHOLD):
                exclusion_df = append_exclusion_row(
                    exclusion_df, subject_id, 'mismatch_3.0back_omission_rate', value, OMISSION_RATE_THRESHOLD
                )

    #sort by subject_id
    if len(exclusion_df) != 0:
        exclusion_df = sort_subject_ids(exclusion_df)
    return exclusion_df

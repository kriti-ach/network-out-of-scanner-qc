import pandas as pd
import glob

folder_path = "/oak/stanford/groups/russpold/data/network_grant/behavioral_data/out_of_scanner/"

for subject_folder in glob.glob(folder_path + "*"):
    subject_id = subject_folder.split("/")[-1]
    print(subject_id)
    for file in glob.glob(subject_folder + "*.csv"):
        df = pd.read_csv(file)
        print(df.head())
        break
    break
    break







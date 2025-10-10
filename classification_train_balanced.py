# resample_splits.py
# Balance classification train set by matching all classes to the count of 'Active'.
# Uses the user's sampling pattern and leaves regression/test splits unchanged.
import os
import pandas as pd

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("outputs/classification_train.csv")
print(df['mainmode'].value_counts())
num_active_mode = (df['mainmode'] == 'Active').sum()
print(num_active_mode)

subsample_private = df[df['mainmode'] == 'Private'].sample(n=num_active_mode, random_state=20008)     
subsample_public = df[df['mainmode'] == 'Public'].sample(n=num_active_mode, replace=True, random_state=20008)   
subsample_hired = df[df['mainmode'] == 'Hired'].sample(n=num_active_mode, replace=True, random_state=20008)   
subsample_other = df[df['mainmode'] == 'Other'].sample(n=num_active_mode, replace=True, random_state=20008)   

balanced_df = df[df['mainmode'] == 'Active']
balanced_df = pd.concat([
    balanced_df, subsample_hired, subsample_other, 
    subsample_private, subsample_public
    ])

print(balanced_df['mainmode'].value_counts())

balanced_df.to_csv(os.path.join("outputs", "classification_train_balanced.csv"), index=False)
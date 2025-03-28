# Description: This script is used to calculate the size of the model.
# The model size is calculated by saving the model to a temporary file and then calculating the size of the file.   
# The model size is then saved in a dictionary with the filename as the key and the size as the value.
# The dictionary is then used to create a dataframe with the filename and the size of the model.    
# The dataframe is then used to calculate the size difference between the original model and the quantized model.
# The result is saved in a csv file.

import pandas as pd
import re
import glob
import os
import torch


def size_of_model(model):
    state = model.state_dict() if hasattr(model, 'state_dict') else model
    temp_file = "temp.p"
    torch.save(state, temp_file)
    size = os.path.getsize(temp_file) / 1e6  # Convert to MB
    os.remove(temp_file)
    return size


folder_path = r'E:\Project_1_QGNN\Barcelo_code - noquantazation\src\logging\results'

file_pattern = os.path.join(folder_path, "*.log")
all_files = glob.glob(file_pattern)


model = "acgnn"
agg_abr = "S"
read_abr = "S"
comb_abr = "T"


pattern = (
    r"(?P<key>p[1-3])-0-0-"
    + re.escape(model) +
    r"-agg" + re.escape(agg_abr) +
    r"-read" + re.escape(read_abr) +
    r"-comb" + re.escape(comb_abr) +
    r"-cl(?P<comb_layers>[0-2])-L(?P<l>\d+)\.log"
)
regex = re.compile(pattern)

data = []

for file in all_files:
    base_name = os.path.basename(file)
    match = regex.fullmatch(base_name)
    if match:
        params = match.groupdict() 
        with open(file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        last_line = lines[-1] if lines else None

        entry = {
            "file": base_name,
            "last_line": last_line
        }
        entry.update(params)
        data.append(entry)

df = pd.DataFrame(data)


split_cols = df['last_line'].str.split(',', expand=True)
split_cols.columns = [
    'train_loss', 'test1_loss', 'test2_loss',
    'train_micro', 'train_macro', 'test1_micro',
    'test1_macro', 'test2_micro', 'test2_macro','elapsed_time'
]
for col in split_cols.columns:
    split_cols[col] = split_cols[col].str.strip().astype(float)

df = pd.concat([df, split_cols], axis=1)
df=df.drop(columns=['last_line'])
print(df)

keys = ['p1', 'p2', 'p3']
model_sizes = {}
for key in keys:
    model_dir = rf'E:\Project_1_QGNN\Barcelo_code - noquantazation\src\saved_models\results\{key}'
    for filename in os.listdir(model_dir):
        if filename.endswith('.pth'):
            model_path = os.path.join(model_dir, filename)
            model = torch.load(model_path, map_location='cpu')
            size = size_of_model(model)
            print(f'{filename}: {size:.2f} MB')
            filename = filename.replace('MODEL', key)
            filename = filename.replace('-acgnn-0', '-0-0-acgnn')
            filename = filename.replace('-H64.pth', '.log')
            model_sizes[filename] = size
print('model size',model_sizes)
df['size_of_model'] = df['file'].map(model_sizes)
df=df.sort_values(by=['key','comb_layers']).reset_index(drop=True)
df=df.drop(columns=['file'])
print(df)
df.to_csv("non_qua_output_tr_relu_acgnn.csv", index=False)
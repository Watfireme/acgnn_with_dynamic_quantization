# This file is used to collect the size of the models generated after quantization
# The result is the size of the original model, the size of the quantized model, and the difference between the two sizes
# The result is saved in the model_sizes_original_dyn_qua_pytorch_all.csv file

import pandas as pd
import os
import torch

keys = ['p1', 'p2', 'p3']
model_sizes = {}
for key in keys:
    model_dir = rf'E:\Project_1_QGNN\Barcelo_code - quantaization - ptq\GNN-logic-master\src\saved_models\results\{key}'
    for filename in os.listdir(model_dir):
        if filename.endswith('.pth'):
            model_path = os.path.join(model_dir, filename)
            model = torch.load(model_path, map_location='cpu')
            size = os.path.getsize(model_path) / 1e6  # Convert to MB
            print(f'{filename}: {size:.2f} MB')
            filename = filename.replace('MODEL', key)
            filename = filename.replace('-acgnn-0', '-0-0-acgnn')
            filename = filename.replace('-acrgnn-0', '-0-0-acrgnn')
            filename = filename.replace('-acrgnn-single-0', '-0-0-acrgnn-single')
            filename = filename.replace('-H64.pth', '.log')
            filename = filename.replace('-H64-quantized.pth', '-quantized.log')
            model_sizes[filename] = size
print('model size',model_sizes)

df = pd.DataFrame(model_sizes.items(), columns=['file', 'size'])
df_quantized = df[df['file'].str.endswith('-quantized.log')].reset_index(drop=True)
df = df[~df['file'].str.endswith('-quantized.log')].reset_index(drop=True)
df_quantized['file'] = df_quantized['file'].str.replace('-quantized.log', '.log')
df=df.join(df_quantized.set_index('file'), on='file', rsuffix='_quantized')
df['size_diff'] = df['size'] - df['size_quantized']
df.to_csv("model_sizes_original_dyn_qua_pytorch_all.csv", index=False)
print('done')
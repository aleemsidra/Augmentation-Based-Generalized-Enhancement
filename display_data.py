"""## **1. Image count statistics**

Dataset contains 6,432 images of seggreated into three different types: 

- Covid-19 patient x-ray image (code 0)
- Normal (healthy) person x-ray image (code 1)
- Pneumonia patient x-ray image (code 2)
"""

import pandas as pd
import os

def show_dataset(data_path, splits, categories):
    df = pd.DataFrame(columns = ['code'] + splits, index = categories)
    for row in categories:
        for col in splits:
            df.loc[row,col] = int(len(os.listdir(os.path.join(data_path, col+'/'+row))))
    df['total'] = df.sum(axis=1).astype(int)
    df.loc['TOTAL'] = df.sum(axis=0).astype(int)
    df['code'] = ['0', '1', '2','']
    print(df)
    return df

import time
import os
import cv2
import random
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models
from torch import optim
from sklearn.metrics import *
from sklearn import metrics
from PIL import Image
import shutil
import pdb
from random import randrange

from display_data import *
from load_data import *
from initialize_model import *
from train_test_modules import *
from evaluation import *
from train_test_modules import *

import argparse

parser = argparse.ArgumentParser(description='covid')
parser.add_argument('--path', default='document/covid', type=str, required=True)
parser.add_argument('--model', default='resnet18', type=str, required=True)
parser.add_argument('--model_name', default='resnet18_enc1', type=str, required=True)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--lr', default=1e-2, type=float)
args = parser.parse_args()


num_workers = 4 #change this parameter based on your system configuration
batch_size = 16 #change this parameter based on your system configuration
seed = 24
random.seed(seed)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
categories = ['COVID19', 'NORMAL', 'PNEUMONIA']
num_classes = len(categories)
splits = ['train', 'val', 'test']



"""## **1. Image count statistics**

Dataset contains 6,432 images of seggreated into three different types: 

- Covid-19 patient x-ray image (code 0)
- Normal (healthy) person x-ray image (code 1)
- Pneumonia patient x-ray image (code 2)
"""

DATA_PATH = args.path
MODEL_PATH = '/home/sidra/Documents/project_2/prob_model/'
MODEL_CSV_PATH = '/home/sidra/Documents/project_2/prob_model_csv/'


df_dataset = show_dataset(DATA_PATH, splits, categories)


'''## **2. Data Loader'''

train_data, train_loader, val_data, val_loader, test_data, test_loader = load_data(DATA_PATH, num_workers, batch_size )
dataset = torch.utils.data.ConcatDataset([train_data, val_data, test_data])


"""## **3. GPU**

If available, enable GPU capacity for model building exercise 
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'You are using {device}')


"""## **4. Define the supporting modules**

### **4.1 Calculate metrics, plot loss graph and create confusion matrix**
DISCUSS***'''


### **5.2 Pre-trained model example: VGG19**

###Pass model name as parameter to run pre-trained model of your preference.


vgg_model = initialize_model(device, args.model, num_classes, use_pretrained=True)

"""### **9.3 Train the  model**"""

vgg_model = initialize_model(device, args.model, num_classes, use_pretrained=True)
np.random.seed(seed)
torch.manual_seed(seed)
model_name = args.model_name
n_epochs = args.epoch
learning_rate = args.lr
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(vgg_model.parameters(), lr=learning_rate)
use_scheduler = False 
mdl_path = os.path.join(MODEL_PATH, f'p_{model_name}.pth')

#Training
pretrained_model, df_vgg_epochs, best_val = train_val_model(vgg_model, model_name, mdl_path,df_dataset, n_epochs, train_loader, val_loader, optimizer)
df_vgg_epochs.to_csv(MODEL_CSV_PATH+args.model_name+"_"+ str(p)+'_.csv', index='False')

#Loading best saved model
vgg_model = initialize_model(device, args.model, num_classes, use_pretrained=False)
dst = torch.load(mdl_path)
vgg_model.load_state_dict(dst)

df = test_model(vgg_model, test_loader)
df.to_csv(MODEL_CSV_PATH+args.model_name+"_"+'test_avg.csv', index='False')






# df = test_model_clean(vgg_model, val_loader, name = 'Validation Clean')
# df.to_csv(MODEL_CSV_PATH+args.model_name+'val_clean.csv', index='False')
# df = test_model(vgg_model, val_loader, name  = 'Validation Avergae')
# df.to_csv(MODEL_CSV_PATH+args.model_name+'val_avg.csv', index='False')

# df = test_model_clean(vgg_model, test_loader )
# df.to_csv(MODEL_CSV_PATH+args.model_name+'test_clean.csv', index='False')
# df = test_model(vgg_model, test_loader)
# df.to_csv(MODEL_CSV_PATH+args.model_name+'test_avg.csv', index='False')

#test_inception_results.to_csv(MODEL_CSV_PATH+'test_'+args.model_name+'test.csv', index='False')




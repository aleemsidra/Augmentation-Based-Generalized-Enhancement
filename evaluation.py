"""## **4. Define the supporting modules**

### **4.1 Calculate metrics, plot loss graph and create confusion matrix**
"""
import numpy as np
import pandas as pd
from sklearn.metrics import *


metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
def calc_metrics(predictions, actuals, row):
    df = pd.DataFrame(columns =metrics)

    Y_pred = predictions
    Y_test = actuals
    df.loc[row, 'Accuracy'] = accuracy_score(Y_test, Y_pred)
    df.loc[row, 'Precision'] = precision_score(Y_test, Y_pred, average="macro")
    df.loc[row, 'Recall'] = recall_score(Y_test, Y_pred, average="macro")
    df.loc[row, 'F1-score'] = f1_score(Y_test, Y_pred, average="macro")
    return df

def create_confusion_matrix(preds, y_test):
    ylist, predlist = [], []
    for pred in preds:
        for item in pred:
            predlist.append(int(item))
    for y in y_test:
        for item in y:
            ylist.append(int(item))
    data_dict = {'y_Actual':    ylist, 'y_Predicted': predlist}
    df = pd.DataFrame(data_dict, columns=['y_Actual','y_Predicted'])
    cm = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['ACTUAL'], colnames=['PREDICTED'])
    return cm



'''Calculate average predictions'''
def avg_prediction(pred_list):
  avg_pred = np.array([sum(x)/len(x) for x in zip(*pred_list)])
  return avg_pred

def plot_train_val_losses(df):
    df2 = pd.melt(df, id_vars=['epoch'], value_vars=['train', 'val'], var_name='process', value_name='loss')
    sns.lineplot(x = "epoch", y = "loss", data = df2, hue = "process",
                style = "process", palette = "hot", dashes = False, 
                markers = ["o", "<"],  legend="brief").set_title("Train and Validation Losses by Epoch")
    plt.show()
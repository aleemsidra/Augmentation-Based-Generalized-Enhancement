# '''Train and Validation'''   
from evaluation import *
from sklearn.model_selection import KFold
import pandas as pd
import torch
import torch.optim as optim
import time
import numpy as np
from random import randrange
from torch import optim
import torch.nn as nn
from sklearn.metrics import *
from sklearn import metrics

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'You are using {device}')


best_acc=0
def train_val_model(model, model_name, modl_path, df_dataset, n_epochs, train_loader, val_loader, optimizer):
    t_start = time.time()
    global best_val_model
    global best_val_loss
    best_val_loss = 1
    global best_val_epoch
    global best_acc
    best_acc = 0
    df = pd.DataFrame(columns = ['model_name', 'epoch', 'train', 'val', 'val_accuracy', 'best_accuracy'])
    print(f"Training model {model_name} with {df_dataset.loc['TOTAL', 'train']} samples and max of {n_epochs} epochs, and validating with {df_dataset.loc['TOTAL', 'val']} samples\n")
    train_size, val_size = len(train_loader), len(val_loader)
    train_loss = []
    for epoch in range(1, n_epochs+1):
        # Beginning of training step
        t0 = time.time()
        model.train()
        train_loss, val_loss = 0.0, 0.0
        alpha = [1.15, 1.1, 1.2, 1.25, 1.3,1.35]
        beta = [-0.1,-0.2,-0.4, 0.1, 0.2, 0.4] #0.2, 0.4, 0.5, 0.7] #,1.1,1.2,1.4, 1.5]
        random_index = randrange(len(alpha))
        r_alpha = alpha[random_index]
        random_index = randrange(len(beta))
        r_beta = beta[random_index]        
        print("alpha: ", r_alpha, " beta: ", r_beta)
        for i, (data, target) in enumerate(train_loader):

        #r = np.random.rand(1)
           # if r < prob:
               # target = target.to(device)
               # data = data.to(device)
           # else:    
            enc_data =  data.numpy()
            enc = np.clip(r_alpha*enc_data+r_beta, 0,255)
            data = torch.from_numpy(enc)
            target = target.to(device)
            data = data.to(device)
 
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().cpu().numpy()

        # Beginning of evaluation step

        model.eval()
        predictions, actuals, val_loss = [], [] , 0
        for j, (data, target) in enumerate(val_loader):
            enc_data =  data.numpy()
            enc = np.clip(r_alpha*enc_data+r_beta, 0,255)
            data = torch.from_numpy(enc)
            target = target.to(device)
            data = data.to(device)
            outputs = model(data)

            loss = criterion(outputs, target)
            val_loss += loss.detach().cpu().numpy()
            actuals.extend(target.cpu().numpy())
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
        
        actuals = np.array(actuals)
        predictions = np.array(predictions)
        accuracy = accuracy_score(actuals, predictions)

        #save best model

        print('accuracy' , accuracy)
        if best_acc < accuracy:
          best_acc = accuracy
          torch.save(model.state_dict(), modl_path)
        print( f'accuracy: {accuracy}  '
                f'best: {best_acc}  ')    
        print(f"Epoch {epoch}:\t train loss={train_loss/train_size:.5f} \t val loss={val_loss/val_size:.5f} \t time={(time.time() - t0):.2f}s")
        df.loc[len(df)] = [model_name, epoch, train_loss/train_size, val_loss/val_size, accuracy, best_acc]

    print(f"Total time training and evaluating: {(time.time()-t_start):.2f}s")
    return model, df

'''Test code'''
'''Test: clean images'''

def test_model_clean(model,loader, name = 'Test Results'):
 predictions, actuals,  pred  = [], [], []
 model.eval()
 print(f"Testing started\n")
 with torch.no_grad():
    for i, (data, target) in enumerate(loader):
        temp_batch =  data.numpy()                        
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        probs  = output.cpu().numpy()
        actuals.extend(target.cpu().numpy())
        predictions.extend(probs.argmax(axis=1))
    actuals = np.array(actuals)
    predictions= np.array(predictions)
    test_accuracy = accuracy_score(actuals, predictions)
    df_test = calc_metrics(predictions, actuals, name).astype(float)
    return  df_test  

'''Test: enhanced images'''

def test_model(model,loader, name = 'Test Results'):
 predictions, actuals,  pred  = [], [], []
 print(type(actuals))
 model.eval()
 print(f"Testing started\n")

 with torch.no_grad():
    for i, (data, target) in enumerate(loader):
        pred = []
        temp_batch =  data.numpy()                        
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        probs  = output.cpu().numpy()
        pred.append(probs)
                    
        '''Random alpha beta'''
        alpha = [1.15, 1.1, 1.2, 1.25, 1.3,1.35]
        beta  = [-0.1,-0.2,-0.4, 0.1, 0.2, 0.4]         
        for j in alpha:
            for k in beta:
            	'''print('alpha beta', '(', j, k , ')') '''       
            	enc_batch = np.clip(j*temp_batch+k, 0,255)
            	enc_batch =torch.from_numpy(enc_batch)
            	enc_batch = enc_batch.to(device)
            	enc_output = model(enc_batch)
            	enc_probs  = enc_output.cpu().numpy()
            	pred.append(enc_probs) 
            	avg_pred = avg_prediction(pred)           	
            	actuals.extend( target.cpu().numpy())
            	predictions.extend(avg_pred.argmax(axis=1))            	
  
    actuals = np.array(actuals)
    predictions= np.array(predictions)
    print('actuals: ', actuals.shape)
    print('pred: ', predictions.shape)
    test_accuracy = accuracy_score(actuals, predictions)
    df_test = calc_metrics(predictions, actuals, name).astype(float)
    print(df_test)
    return  df_test  

"""## **10 K-Fold Cross Validation**

Use k-fold cross validation technique to evaludate models as follows:

1.   Set k=5 to perform cross validation using 5 folds.
2.   Generate train and test data using random sampler for each fold.
1.   Perform forward propagation using pre-trained model
2.   Calculate train loss and perform optimization with zero gradient
1.   Evaluation the model using test data
2.   Calculate accuracy measures such as accuracy score, precision, recall and F1-score.
"""


def train_val_model_kfold(fold, model, train_loader_kfold, k_fold_mdl_path, model_name, n_epochs, optimizer,test_kfold_loader):
    
    t_start = time.time()
    global best_acc
    best_acc = 0
    df = pd.DataFrame(columns = ['model_name', 'epoch', 'train'])
    print(f"Training model {model_name } , k_fold : {fold+1} ") 
    train_size = len(train_loader_kfold)
    for epoch in range(1, n_epochs+1):
        predictions, actuals,  pred  = [], [], []
        # Beginning of training step
        t0 = time.time()
        #model.train()
        train_loss = 0.0
        alpha = [1.15, 1.1, 1.2, 1.25, 1.3,1.35]
        beta = [-0.1,-0.2,-0.4, 0.1, 0.2, 0.4] 
        random_index = randrange(len(alpha))
        r_alpha = alpha[random_index]
        random_index = randrange(len(beta))
        r_beta = beta[random_index]        
        #print("alpha: ", r_alpha, " beta: ", r_beta)
        for i, (data, target) in enumerate(train_loader_kfold):
            enc_data =  data.numpy()
            enc = np.clip(r_alpha*enc_data+r_beta, 0,255)
            data = torch.from_numpy(enc)
            target = target.to(device)
            data = data.to(device)
 
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().cpu().numpy()
            actuals.extend(target.cpu().numpy())
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
        
        actuals = np.array(actuals)
        predictions = np.array(predictions)
        accuracy = accuracy_score(actuals, predictions)
        actuals, predictions  = k_fold_test_model(fold, model,test_kfold_loader, model_name, name = 'Test Results')
        #save best model
        actuals = np.array(actuals)
        predictions = np.array(predictions)
        accuracy = accuracy_score(actuals, predictions)
        print('accuracy' , accuracy)
          
        if best_acc < accuracy:
          best_acc = accuracy
          torch.save(model.state_dict(), k_fold_mdl_path)
        print( f'accuracy: {accuracy}  '
                f'best: {best_acc}  ')  


        print(f"Epoch {epoch}:\t train loss={train_loss/train_size:.5f} \t time={(time.time() - t0):.2f}s")
        df.loc[len(df)] = [model_name, epoch, train_loss/train_size]
    #print(f"Total time training and evaluating: {(time.time()-t_start):.2f}s")
    return model, df

def k_fold_test_model(fold, model,loader, model_name, name = 'Test Results'):
 predictions, actuals,  pred  = [], [], []
 model.eval()
 print(f"Testing model {model_name } , k_fold : {fold+1} ")
 with torch.no_grad():
    for i, (data, target) in enumerate(loader):
        pred = []
        temp_batch =  data.numpy()                        
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        probs  = output.cpu().numpy()
        pred.append(probs)
                    
        '''Random alpha beta'''
        alpha = [1.15, 1.1, 1.2, 1.25, 1.3,1.35]
        beta  = [-0.1,-0.2,-0.4, 0.1, 0.2, 0.4]         
        for j in alpha:
            for k in beta:
            	'''print('alpha beta', '(', j, k , ')') '''       
            	enc_batch = np.clip(j*temp_batch+k, 0,255)
            	enc_batch =torch.from_numpy(enc_batch)
            	enc_batch = enc_batch.to(device)
            	enc_output = model(enc_batch)
            	enc_probs  = enc_output.cpu().numpy()
            	pred.append(enc_probs) 
        avg_pred = avg_prediction(pred)           	
        actuals.extend( target.cpu().numpy())
        predictions.extend(avg_pred.argmax(axis=1))            	

    return  predictions, actuals 


def validate_model_kfold(model, model_name, k_folds, k_fold_mdl_path, dataset,batch_size, num_workers, n_epochs, optimizer):
    #model.eval()
    print(f"Validating the model {model_name} with {k_folds}-fold \n")
    df = pd.DataFrame(columns = metrics)
    kfold = KFold(n_splits=k_folds, shuffle=True)
    print(len(dataset))

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        k_fold_mdl_path = k_fold_mdl_path.replace(".pth",str(fold))+".pth"
        train_kfold_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  num_workers=num_workers, sampler=train_subsampler)
        test_kfold_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=test_subsampler)
        model, df_vgg_epochs = train_val_model_kfold(fold, model, train_kfold_loader, k_fold_mdl_path, model_name, n_epochs, optimizer,test_kfold_loader)
        dst = torch.load(k_fold_mdl_path)
        model.load_state_dict(dst)

        predictions, actuals = k_fold_test_model(fold, model, test_kfold_loader,  model_name, name = 'k_fold Test Results',)

        df_aux = calc_metrics(predictions, actuals, 'FOLD '+str(fold+1))
        df = df.append(df_aux)      
    df.loc['Average'] = df.mean(axis=0)
    print(df.astype(float))
    return df
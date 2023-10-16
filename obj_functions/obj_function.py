import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from numpy import genfromtxt
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")
import torch


import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from numpy import genfromtxt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch


# class XGBoost:
#     def __init__(self,task,seed=1):
#         # define the search range for each variable
#         self.bounds = torch.tensor(np.asarray([
#                                 [0.,10.],  # alpha
#                                   [0.,10.],# gamma 
#                                   [5.,15.], #max_depth
#                                   [1.,20.],  #min_child_weight
#                                   [0.5,1.],  #subsample
#                                   [0.1,1] #colsample
#                                  ]).T)
            
#         self.dim = 6
#         self.fstar = 100
#         self.task = task

#         self.seed= seed
        
#         if task == 'skin':
#             self.data = np.genfromtxt('Skin_NonSkin.txt', dtype=np.int32)
        
#     def __call__(self, X): # this is actually a Branin function
        
#         X = X.numpy().reshape(6,)

        
#         alpha,gamma,max_depth,min_child_weight,subsample,colsample=X[0],X[1],X[2],X[3],X[4],X[5]
        
#         #data = np.genfromtxt('Skin_NonSkin.txt', dtype=np.int32)

#         if self.task == 'skin':
#             outputs = self.data[:,3]
#             inputs = self.data[:,0:3]
#             X_train1, X_test1, y_train1, y_test1 = train_test_split(inputs, outputs, test_size=0.85, random_state=self.seed)
#             y_train1 = y_train1-1
            
#             reg = XGBClassifier(reg_alpha=alpha, gamma=gamma, max_depth=int(max_depth), subsample=subsample, 
#                         min_child_weight=min_child_weight,colsample_bytree=colsample, n_estimators = 2, random_state=self.seed, objective = 'binary:logistic', booster='gbtree',eval_metric='logloss',silent=None)
#             score = np.array(cross_val_score(reg, X=X_train1, y=y_train1).mean())
      
#         return torch.tensor([score*100])
    
    
class XGBoost:
    def __init__(self,task,seed=1):
        # define the search range for each variable
        self.bounds = torch.tensor(np.asarray([
                                [0.,10.],  # alpha
                                  [0.,10.],# gamma 
                                  [5.,15.], #max_depth
                                  [1.,20.],  #min_child_weight
                                  [0.5,1.],  #subsample
                                  [0.1,1] #colsample
                                 ]).T)
            
        self.dim = 6
        self.fstar = 100

        self.seed= seed
        
        self.task = task
        
        if task == 'skin':
            data = np.genfromtxt('Skin_NonSkin.txt', dtype=np.int32)
            outputs = data[:,3]
            inputs = data[:,0:3]
            X_train1, X_test1, y_train1, y_test1 = train_test_split(inputs, outputs, test_size=0.85, random_state=self.seed)
            y_train1 = y_train1-1
            
            self.X_train1 = X_train1
            self.y_train1 = y_train1
            
        elif task == 'bank':
            data = pd.read_csv('BankNote_Authentication.csv')
            X = data.loc[:, data.columns!='class']
            y = data['class']
            outputs = np.array(y)
            inputs = np.array(X)
            X_train1, X_test1, y_train1, y_test1 = train_test_split(inputs, outputs, test_size=0.1, random_state=self.seed)
            
            self.X_train1 = X_train1
            self.y_train1 = y_train1
        elif task == 'iris':
            df_data = pd.read_csv('Iris.csv')
            class_mapping = {'Iris-setosa':0, 'Iris-versicolor':1,'Iris-virginica':2}
            df_data['Species'] = df_data['Species'].map(class_mapping)
            df_data = df_data.drop(axis=1,columns=['Id'])
            
            X = df_data.loc[:, df_data.columns!='Species']
            y = df_data['Species']
            
            outputs = np.array(y)
            inputs = np.array(X)
            X_train1, X_test1, y_train1, y_test1 = train_test_split(inputs, outputs, test_size=0.01, random_state=1)
            
            self.X_train1 = X_train1
            self.y_train1 = y_train1
            
        elif task == 'breast':
            df_data = pd.read_csv('Breast_Cancer_Wisconsin.csv')
            df_data = df_data.drop(axis=1,columns=['id'])
            class_mapping = {'M':0, 'B':1}
            df_data['diagnosis'] = df_data['diagnosis'].map(class_mapping)
            
            X = df_data.loc[:, df_data.columns!='diagnosis']
            y = df_data['diagnosis']

            outputs = np.array(y)
            inputs = np.array(X)
            X_train1, X_test1, y_train1, y_test1 = train_test_split(inputs, outputs, test_size=0.05, random_state=1)

            self.X_train1 = X_train1
            self.y_train1 = y_train1

            
    def __call__(self, X): # this is actually a Branin function
        
        X = X.numpy().reshape(6,)

        
        alpha,gamma,max_depth,min_child_weight,subsample,colsample=X[0],X[1],X[2],X[3],X[4],X[5]
        
        if self.task == 'bank'  or self.task == 'skin' or self.task == 'breast':
            reg = XGBClassifier(reg_alpha=alpha, gamma=gamma, max_depth=int(max_depth), subsample=subsample, 
                        min_child_weight=min_child_weight,colsample_bytree=colsample, n_estimators = 2, random_state=self.seed, objective = 'binary:logistic', booster='gbtree',eval_metric='logloss',silent=None)
            score = np.array(cross_val_score(reg, X=self.X_train1, y=self.y_train1).mean())
        elif self.task == 'iris':
            reg = XGBClassifier(reg_alpha=alpha, gamma=gamma, max_depth=int(max_depth), subsample=subsample, 
                    min_child_weight=min_child_weight,colsample_bytree=colsample, n_estimators = 2, random_state=1, objective = 'multi:softmax', booster='gbtree',eval_metric='logloss',silent=None)
            score = np.array(cross_val_score(reg, X=self.X_train1, y=self.y_train1).mean())
      
        return torch.tensor([score*100])
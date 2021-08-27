#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 10:28:45 2021

@author: victorquere
"""

import os
os.chdir(r'C:\Users\victo\Desktop\Thèse\code final')

#imports bibliothèques
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Import des données
df_train = pd.read_csv('train_comp.csv',sep=';',header=0, na_values = 'nan')
df_test =pd.read_csv('test_comp.csv',sep=';',header=0, na_values = 'nan')

#Suppression des lignes HEMO_SS_TENT et HEMO_PROF
df_train = df_train.drop(['HEMO_SS_TENT', 'HEMO_PROF'], axis = 1)
df_test = df_test.drop(['HEMO_SS_TENT', 'HEMO_PROF'], axis = 1)

#modification type des données df_train
df_train['SEXE'] = df_train['SEXE'].astype(str)
df_train['ATCD_ARYTH'] = df_train['ATCD_ARYTH'].astype(str)
df_train['ATCD_HTA'] = df_train['ATCD_HTA'].astype(str)
df_train['ATCD_DIAB'] = df_train['ATCD_DIAB'].astype(str)
df_train['ATCD_DYSLIP'] = df_train['ATCD_DYSLIP'].astype(str)
df_train['ATCD_RANKIN'] = df_train['ATCD_RANKIN'].astype(str)
df_train['LACUNE'] = df_train['LACUNE'].astype(str)
df_train['ENTREE_HYPER'] = df_train['ENTREE_HYPER'].astype(str)
df_train['ENTREE_AGG'] = df_train['ENTREE_AGG'].astype(str)
df_train['ENTREE_STATINE'] = df_train['ENTREE_STATINE'].astype(str)
df_train['ENTREE_FIBRATE'] = df_train['ENTREE_FIBRATE'].astype(str)
df_train['ENTREE_PARA'] = df_train['ENTREE_PARA'].astype(str)
df_train['ENTREE_anticoagulant'] = df_train['ENTREE_anticoagulant'].astype(str)
df_train['atcd_arthero'] = df_train['atcd_arthero'].astype(str)
df_train['statut_j30_g'] = df_train['statut_j30_g'].astype(str)

#modification type de données df_test
df_test['SEXE'] = df_test['SEXE'].astype(str)
df_test['ATCD_ARYTH'] = df_test['ATCD_ARYTH'].astype(str)
df_test['ATCD_HTA'] = df_test['ATCD_HTA'].astype(str)
df_test['ATCD_DIAB'] = df_test['ATCD_DIAB'].astype(str)
df_test['ATCD_DYSLIP'] = df_test['ATCD_DYSLIP'].astype(str)
df_test['ATCD_RANKIN'] = df_test['ATCD_RANKIN'].astype(str)
df_test['LACUNE'] = df_test['LACUNE'].astype(str)
df_test['ENTREE_HYPER'] = df_test['ENTREE_HYPER'].astype(str)
df_test['ENTREE_AGG'] = df_test['ENTREE_AGG'].astype(str)
df_test['ENTREE_STATINE'] = df_test['ENTREE_STATINE'].astype(str)
df_test['ENTREE_FIBRATE'] = df_test['ENTREE_FIBRATE'].astype(str)
df_test['ENTREE_PARA'] = df_test['ENTREE_PARA'].astype(str)
df_test['ENTREE_anticoagulant'] = df_test['ENTREE_anticoagulant'].astype(str)
df_test['atcd_arthero'] = df_test['atcd_arthero'].astype(str)
df_test['statut_j30_g'] = df_test['statut_j30_g'].astype(str)

#vérification type et dimension des données
print (df_train.dtypes)
print ('Dimensions des données de train :', df_train.shape, 'Dimensions des données de test :', df_test.shape)

#############################################################################
                                #Pre-processing
#############################################################################
#Normalisation des données non nécessaire concernant les algorithmes utilisés

#création X et Y
x_train = df_train.drop(['statut_j30_g'], axis=1)
y_train = df_train['statut_j30_g']

x_test = df_test.drop(['statut_j30_g'], axis=1)
y_test = df_test['statut_j30_g']

#############################################################################
                           #Modelisation RF
#############################################################################
from sklearn.model_selection import GridSearchCV, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score

RF = RandomForestClassifier(random_state=42, class_weight='balanced_subsample')
#############################################################################
                          #Validation curve
#############################################################################


#### n_estimators ####
param_range = [100,500,1000,1500,2000]

train_scoreNum, test_scoreNum = validation_curve(
                                RF,
                                X = x_train, y = y_train, 
                                param_name = 'n_estimators', 
                                param_range = param_range, cv = 5,
                                scoring="roc_auc", n_jobs= -1)

train_scoreNum_mean = np.mean(train_scoreNum, axis=1)
train_scoreNum_std = np.std(train_scoreNum, axis=1)
test_scoreNum_mean = np.mean(test_scoreNum, axis=1)
test_scoreNum_std = np.std(test_scoreNum, axis=1)
plt.ylim(0.0, 1.1)
lw = 2

plt.title("Validation Curve n_estimators")
plt.xlabel("n_estimators")
plt.ylabel("Score")
plt.semilogx(param_range, train_scoreNum_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scoreNum_mean - train_scoreNum_std,
                 train_scoreNum_mean + train_scoreNum_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scoreNum_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scoreNum_mean - test_scoreNum_std,
                 test_scoreNum_mean + test_scoreNum_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


#### max_depth ####
param_range = [3,6,9,12,15,20,25,30]

train_scoreNum, test_scoreNum = validation_curve(
                                RF,
                                X = x_train, y = y_train, 
                                param_name = 'max_depth', 
                                param_range = param_range, cv = 5,
                                scoring="roc_auc", n_jobs= -1)

train_scoreNum_mean = np.mean(train_scoreNum, axis=1)
train_scoreNum_std = np.std(train_scoreNum, axis=1)
test_scoreNum_mean = np.mean(test_scoreNum, axis=1)
test_scoreNum_std = np.std(test_scoreNum, axis=1)
#plt.ylim(0.0, 1.9)
lw = 2

plt.title("Validation Curve max_depth")
plt.xlabel("max_depth")
plt.ylabel("Score")
plt.semilogx(param_range, train_scoreNum_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scoreNum_mean - train_scoreNum_std,
                 train_scoreNum_mean + train_scoreNum_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scoreNum_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scoreNum_mean - test_scoreNum_std,
                 test_scoreNum_mean + test_scoreNum_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


#### min_samples_leaf ####
param_range = [2,4,6,8,10,12,14,16,18,20,30,40,50,60]

train_scoreNum, test_scoreNum = validation_curve(
                                RF,
                                X = x_train, y = y_train, 
                                param_name = 'min_samples_leaf', 
                                param_range = param_range, cv = 5,
                                scoring="roc_auc", n_jobs= -1)

train_scoreNum_mean = np.mean(train_scoreNum, axis=1)
train_scoreNum_std = np.std(train_scoreNum, axis=1)
test_scoreNum_mean = np.mean(test_scoreNum, axis=1)
test_scoreNum_std = np.std(test_scoreNum, axis=1)
#plt.ylim(0.0, 1.9)
lw = 2

plt.title("Validation Curve min_samples_leaf")
plt.xlabel("min_samples_leaf")
plt.ylabel("Score")
plt.semilogx(param_range, train_scoreNum_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scoreNum_mean - train_scoreNum_std,
                 train_scoreNum_mean + train_scoreNum_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scoreNum_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scoreNum_mean - test_scoreNum_std,
                 test_scoreNum_mean + test_scoreNum_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()

#### min_samples_split ####
param_range = [2,5,10,20,30,40,50,60,70,80,90,100]

train_scoreNum, test_scoreNum = validation_curve(
                                RF,
                                X = x_train, y = y_train, 
                                param_name = 'min_samples_split', 
                                param_range = param_range, cv = 5,
                                scoring="roc_auc", n_jobs= -1)

train_scoreNum_mean = np.mean(train_scoreNum, axis=1)
train_scoreNum_std = np.std(train_scoreNum, axis=1)
test_scoreNum_mean = np.mean(test_scoreNum, axis=1)
test_scoreNum_std = np.std(test_scoreNum, axis=1)
#plt.ylim(0.0, 1.9)
lw = 2

plt.title("Validation Curve min_samples_split")
plt.xlabel("min_samples_split")
plt.ylabel("Score")
plt.semilogx(param_range, train_scoreNum_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scoreNum_mean - train_scoreNum_std,
                 train_scoreNum_mean + train_scoreNum_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scoreNum_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scoreNum_mean - test_scoreNum_std,
                 test_scoreNum_mean + test_scoreNum_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()

#############################################################################
                          #Gridsearch
############################################################################# 
param_grid = {'criterion' : ['gini'],
              'bootstrap': [True],
              'max_depth': [None],
              'max_features': ['log2','auto',None],
              'min_samples_leaf': [1],
              'min_samples_split': [46,47,48,49,50,51,52,53],
              'n_estimators': [100]}

grid_searchCV = GridSearchCV (RF, param_grid, scoring='roc_auc', cv = 5, n_jobs = -1, refit=True)
grid_searchCV.fit(x_train, y_train)

grid = grid_searchCV.best_estimator_
print('best score du GridSearch:', grid_searchCV.best_score_)
print('les meilleurs paramêtres du GridSearch sont :', grid_searchCV.best_params_)

#############################################################################
                          #Evaluation des résultats 
#############################################################################
y_pred = grid.predict(x_test)
print ('Résultats sur les données de test :')
print (classification_report(y_test, y_pred))
print (roc_auc_score (y_test, y_pred))
print (confusion_matrix(y_test, y_pred))

#############################################################################
                          #Feature importance 
#############################################################################
feat_importances = pd.Series(grid.feature_importances_, index= x_train.columns)
feat_importances.nlargest(25).plot.bar()
plt.ylabel("Mean decrease Gini")
plt.xlabel("Features")
plt.show()

#############################################################################
                          #Evaluation des résultats 
#############################################################################
from sklearn import metrics

y_probas = grid.predict_proba(x_test)
y_probas = y_probas[:,1]

y_true = pd.DataFrame(data = y_test)
y_true['statut_j30_g'] = y_true['statut_j30_g'].astype(int)

fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probas)

# Print ROC curve
plt.plot(fpr,tpr,label="Complete data, auc= 0.78")
plt.title( 'ROC curve Random Forest on complete Data')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 4)
plt.show()



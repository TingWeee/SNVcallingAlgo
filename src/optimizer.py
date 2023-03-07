random_states = 1
# Classifier/Regressor
from xgboost import XGBClassifier, DMatrix
from functools import partial
import pprint
# Model selection
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
# Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from time import time
# Skopt functions
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Real, Categorical, Integer

# Data processing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import allel
import numpy as np
import pandas as pd

from functools import reduce
import pprint
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import seaborn as sns
from numpy import nan

# Prepare input dataset
df = pd.read_csv("combined_dataset.csv")
df = df.iloc[: , 1:]
print('finish reading dataset')
for col in df.columns:
    if df[col].dtype == 'O':
        df[col] = df[col].astype('category')
enc = OrdinalEncoder()
df[['STATUS_vd']] = enc.fit_transform(df[['STATUS_vd']])

X = df[df.columns[~df.columns.isin(['truth','POS','CHROM'])]]
y = df['truth'] 

# Split into test, train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state = random_states)

def report_performance(optimizer, X, y, title="model", callbacks=None):
    
    start = time()
    print('running_report')
    
    if callbacks is not None:
        optimizer.fit(X, y, callback=callbacks)
    else:
        optimizer.fit(X, y)
        
    d = pd.DataFrame(optimizer.cv_results_)
    
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_
    
    print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "
           + u"\u00B1"+" %.3f") % (time() - start, 
                                   len(optimizer.cv_results_['params']),
                                   best_score,
                                   best_score_std))    
    print('Best parameters:')
    pprint.pprint(best_params)
    print()
    return optimizer

clf = XGBClassifier(n_jobs = -1, 
                    eval_metric = 'logloss',
                    objective='binary:logistic',
                    enable_categorical=True,
                    tree_method='approx', verbosity = 1)

from skopt.space import Real
search_spaces = {'learning_rate': Real(low = 0.01, high = 10, prior ='log-uniform'),
                 'n_estimators': Integer(10, 5000),
                 'max_depth': Integer(2, 20), #'min-child-weight': Integer(1, 5),
                 'colsample_bytree': Real(0.1, 1.0, 'uniform'), # subsample ratio of columns by tree, for samples with lots of features
                 'gamma': Real(1e-9, 100., 'log-uniform'), # Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be
                 'reg_alpha': Real(1e-9, 100., 'log-uniform'), # L1 regularization
                 'reg_lambda': Real(1e-9, 100, 'log-uniform')
   }

baye_opt = BayesSearchCV(
    estimator = clf,
    search_spaces = search_spaces,
    scoring = 'f1',
    cv = StratifiedKFold(n_splits = 7, shuffle = True), # validation strategy
    n_iter=100,                                       # max number of trials
    n_points=5,                                       # number of hyperparameter sets evaluated at the same time
    n_jobs=-1,                                        # number of jobs
    #return_train_score=False,refit = False
    refit = False, verbose = 1,
    # optimizer_kwargs={'base_estimator': 'GP'}, # optmizer parameters: we use Gaussian Process (GP)
    random_state=42)


overdone_control = DeltaYStopper(delta=0.0001)                    # stop if the gain of the optimization becomes too small
time_limit_control = DeadlineStopper(total_time=60*60*6)          # impose a time limit

optimi = report_performance(baye_opt, X_train, y_train,'XGBoost_classifier', 
                          callbacks=[overdone_control, time_limit_control])

from skopt import dump, load
dump(optimi, 'optimise.pkl')

best_params = optimi.best_params_

import csv
with open('best_params_file3.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=best_params.keys())
    writer.writeheader()
    writer.writerow(best_params)


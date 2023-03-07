# Gradient Boosting Algorithm Implementation for accurate identification of SNVs
An example project that predicts SNVs from vcf files from 4 different variant callers (FreeBayes, Mutect2, VarDict, VarScan) using a Gradient Boosted Machine.

## User Guide and Manual
To use the pre-trained models to make predictions, users can refer to src/Model_Testing.ipynb which contains a section on how to load and test pre-trained models. There are 3 models currently available and we recommend using tuned1.model which has the best performance on all datasets so far. Users should use the *getmerged()* function provided to obtain the input dataset or ensure that their columns and object type matches the one shown in the notebook
```
xgb_clf = xgb.XGBClassifier()
xgb_clf.load_model('Models/tuned1.model')
y_pred = xgb_clf.predict(X)
```

To train and optimise models, users can run Optimizer.py script but change input to their own dataset. Users can limit the duration of the optimization by changing time_limit_control. After completion, the file print the optimal parameters (parameters that maximizes the score of held-out data according to the scoring parameter) and save it in 'optimise.pkl' which is the output object of BayesSearchCV(). 
User can then use the optimal hyperparameters to train model as in 'src/GradientBoostingOptimised.ipynb'.
```
bayes_cv = skopt.load('optimise.pkl')
xgb_clf = XGBClassifier(**bayes_cv.best_params_)
xgb_clf.fit(X_train, y_train)
```

## Feature Selection and Dataset Testing
Code for feature selection and combinatorial dataset testing ca be found in Features_selection.ipynb and Dataset_testing.ipynb respectively

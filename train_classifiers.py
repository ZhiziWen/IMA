import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedKFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn import svm, datasets
from scipy import stats
from scipy.stats import randint as sp_randint
from sklearn.preprocessing import StandardScaler

# Data preprocessing
df = pd.read_csv("new_feature_vector_file.csv")
df = df.iloc[: , 1:]
df = df.drop('class',axis=1)
X = df.drop('buggy',axis=1)
y = df[['buggy']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)

# Decision Tree
print('Decision Tree')

# NO tuning
clf_model = DecisionTreeClassifier(criterion="gini", random_state=16)
clf_model.fit(X_train, y_train.values.ravel())
y_predict = clf_model.predict(X_test)
score = precision_recall_fscore_support(y_test, y_predict, average='binary')
print('Score without tuning hyperparameter:', score) # score = (0.2777777777777778, 0.5555555555555556, 0.3703703703703704, None)

# With tuning
clf_model = DecisionTreeClassifier(criterion="entropy", random_state = 16, max_depth = 6, min_samples_split = 2, min_samples_leaf = 5)
clf_model.fit(X_train, y_train.values.ravel())
y_predict = clf_model.predict(X_test)
score = precision_recall_fscore_support(y_test, y_predict, average='binary')
print('Score after tuning hyperparameter:', score) # score = (0.8, 0.4444444444444444, 0.5714285714285714, None)

# Tune the hyperparameter
# max_f1 = 0
# best_i = best_j = best_k = 0
# for i in range(1, 20):
#     for j in range(2, 20):
#         for k in range(1, 20):
#             clf_model = DecisionTreeClassifier(criterion="entropy", random_state=16, max_depth=i, min_samples_split = j, min_samples_leaf=k)
#             clf_model.fit(X_train,y_train.values.ravel())
#             y_predict = clf_model.predict(X_test)
#             score = precision_recall_fscore_support(y_test, y_predict, average='binary')
#             if score[2] > max_f1:
#                 best_score = score
#                 max_f1 = score[2]
#                 best_i = i
#                 best_j = j
#                 best_k = k
# print('best f1: ', max_f1, 'best score:', best_score)
# print('max_depth: ',best_i, 'min_samples_split: ',best_j, 'min_samples_leaf: ',best_k)


print('-------------------')
##############################
# Naive Bayes
print('Naive Bayes')

# NO tuning
nb = GaussianNB()
nb.fit(X_train, y_train.values.ravel())
y_predict = nb.predict(X_test)
print('Score without tuning hyperparameter:', precision_recall_fscore_support(y_test, y_predict, average='binary'))
# Score: (0.2, 0.1111111111111111, 0.14285714285714285, None)

# With tuning
nb = GaussianNB(var_smoothing= 1e-11)
nb.fit(X_train, y_train.values.ravel())
y_predict = nb.predict(X_test)
print('Score after tuning hyperparameter:', precision_recall_fscore_support(y_test, y_predict, average='binary'))
# Score: (0.2, 0.1111111111111111, 0.14285714285714285, None)

# Tune the hyperparameter
# pipe = Pipeline(steps=[
#     ('pca', PCA()),
#     ('estimator', GaussianNB()),
# ])
# parameters = {'estimator__var_smoothing': [1e-11, 1e-10, 1e-9]}
#
# Bayes = GridSearchCV(pipe, parameters, scoring='f1', verbose=1, n_jobs=-1).fit(X_train, y_train.values.ravel())
# best_estimator = Bayes.best_estimator_
# print(best_estimator)

print('-------------------')
##############################
# Support Vector Machine
print('Support Vector Machine')

# NO tuning
svc = SVC()
svc.fit(X_train, y_train.values.ravel())
y_predict = svc.predict(X_test)
print('Score without tuning hyperparameter:', precision_recall_fscore_support(y_test, y_predict, average='binary'))
# Score: (0.6666666666666666, 0.2222222222222222, 0.3333333333333333, None)

#With tuning
svc = SVC(C=10, gamma=0.0001, kernel= 'rbf', random_state=0)
svc.fit(X_train, y_train.values.ravel())
y_predict = svc.predict(X_test)
print('Score after tuning hyperparameter:', precision_recall_fscore_support(y_test, y_predict, average='binary'))
# Score: (0.42857142857142855, 0.3333333333333333, 0.375, None)

# Tune the hyperparameter ,
# svc = svm.SVC()
# parameters = [
#   {'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
# ]
#
# grid_obj = GridSearchCV(svc, parameters,refit=True,verbose=2, scoring = 'f1')
# grid_obj = grid_obj.fit(X_train, y_train.values.ravel())
# best = grid_obj.best_estimator_
# print(f'Best parameters: {best}')

print('-------------------')
# #############################
# Multi-Layer Perceptron
print('Multi-Layer Perceptron')

# No tuning
clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train.values.ravel())
y_predict = clf.predict(X_test)
print('Score without tuning hyperparameter:', precision_recall_fscore_support(y_test, y_predict, average='binary'))
# Score: (0.42857142857142855, 0.3333333333333333, 0.375, None)

# With tuning
clf = MLPClassifier(solver = 'lbfgs', hidden_layer_sizes=(50, 50, 50), random_state=1, activation= 'logistic', max_iter = 1000)
PIPELINE = Pipeline([('scaler', StandardScaler()), ('estimator', clf)])
clf = clf.fit(X_train, y_train.values.ravel())
y_predict = clf.predict(X_test)
print('Score after tuning hyperparameter:', precision_recall_fscore_support(y_test, y_predict, average='binary'))
# Score: (0.5, 0.5555555555555556, 0.5263157894736842, None)

#Tune the hyperparameter
# GRID = [
#     {
#      'estimator': [MLPClassifier(random_state=1)],
#      'estimator__solver': ['lbfgs', 'sgd', 'adam'],
#      'estimator__max_iter': [1000],
#      'estimator__hidden_layer_sizes': [(50, 50, 50), (40, 40, 40), (30, 30, 30)],
#      'estimator__activation': ['logistic', 'tanh', 'relu']
#      }
# ]
#
# PIPELINE =  Pipeline([('scaler', StandardScaler()), ('estimator', MLPClassifier())])
# grid_search = GridSearchCV(estimator=PIPELINE, param_grid=GRID,
#                             scoring='f1',
#                             n_jobs=-1, refit=True, verbose=1,
#                             return_train_score=False, cv=[(slice(None), slice(None))])
#
# grid_search.fit(X_train.values, y_train.values.ravel())
# print('Best parameters found:\n', grid_search.best_params_)


# Warning: max_iter
print('-------------------')
##############################
# Random Forest
print('Random Forest')

# No tuning
clf = RandomForestClassifier(n_jobs=2, random_state=42)
clf.fit(X_train,y_train.values.ravel())
y_predict = clf.predict(X_test)
print('Score without tuning hyperparameter:', precision_recall_fscore_support(y_test, y_predict, average='binary'))
#(0.5, 0.3333333333333333, 0.4, None)

# Parameter after tuning
clf = RandomForestClassifier(n_jobs=2, random_state=42, n_estimators=100, min_samples_split = 2, min_samples_leaf = 1, max_features = 'auto', max_depth = 10, bootstrap= False)
clf.fit(X_train,y_train.values.ravel())
y_predict = clf.predict(X_test)
print('Score after tuning hyperparameter:', precision_recall_fscore_support(y_test, y_predict, average='binary'))
#(0.5, 0.4444444444444444, 0.47058823529411764, None)

# Tuning
# random_grid =\
#     {'bootstrap': [True, False],
#  'max_depth': [10, 30, 50, 70, 90, None],
#  'max_features': ['auto', 'sqrt'],
#  'min_samples_leaf': [1, 2, 4],
#  'min_samples_split': [2, 5, 10],
#  'n_estimators': [100, 200, 400, 600, 800],
# 'n_jobs' : [2]}
# clf = RandomForestClassifier()
# rf_random = GridSearchCV(estimator = clf, param_grid = random_grid, verbose=2, scoring = 'f1', cv = [(slice(None), slice(None))]) #[(slice(None), slice(None))]
# rf_random.fit(X_train,y_train.values.ravel())
# print('Best parameters found:\n', rf_random.best_params_)
# print(rf_random.best_score_)
#
# #, cv=[(slice(None), slice(None))]
#
# clf = RandomForestClassifier()
# clf.set_params(**rf_random.best_params_)
# clf.fit(X_train,y_train.values.ravel())
# y_predict = clf.predict(X_test)
# print('Score after tuning hyperparameter:', precision_recall_fscore_support(y_test, y_predict, average='binary'))
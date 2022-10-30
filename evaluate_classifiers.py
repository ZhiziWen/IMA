from sklearn.model_selection import cross_validate, train_test_split, RepeatedKFold
from scipy.stats import wilcoxon
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.graph_objects as go

df = pd.read_csv("new_feature_vector_file.csv")
df = df.iloc[: , 1:]
df = df.drop('class',axis=1)
X = df.drop('buggy',axis=1)
y = df[['buggy']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)
scoring = {'f1': 'f1', 'precision': 'precision', 'recall': 'recall'}
cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=1)

# dummy
dummy_clf = DummyClassifier(strategy="constant", constant = 1)
results1 = cross_validate(estimator=dummy_clf, X = X, y = y.values.ravel(), scoring=scoring, cv = cv, return_train_score=True)
print("1")

# models
dt_model = DecisionTreeClassifier(criterion="entropy", random_state = 16, max_depth = 6, min_samples_split = 2, min_samples_leaf = 5)
results2 = cross_validate(estimator=dt_model, X = X, y = y.values.ravel(), scoring=scoring, cv = cv, return_train_score=True)
print("2")

nb_model = GaussianNB(var_smoothing= 1e-11)
results3 = cross_validate(estimator=nb_model, X = X, y = y.values.ravel(), scoring=scoring, cv = cv, return_train_score=True)
print("3")

svc = SVC(C=10, gamma=0.0001, kernel= 'rbf', random_state=0)
results4 = cross_validate(estimator=svc, X = X, y = y.values.ravel(), scoring=scoring, cv = cv, return_train_score=True)
print("4")

clf = MLPClassifier(solver = 'lbfgs', hidden_layer_sizes=(50, 50, 50), random_state=1, activation= 'logistic', max_iter = 1000)
PIPELINE = Pipeline([('scaler', StandardScaler()), ('estimator', clf)])
results5 = cross_validate(estimator=PIPELINE, X = X, y = y.values.ravel(), scoring=scoring, cv = cv, return_train_score=True)
print("5")

rfc = RandomForestClassifier(n_jobs=2, random_state=42, n_estimators=100, min_samples_split = 2, min_samples_leaf = 1, max_features = 'auto', max_depth = 10, bootstrap= False)
results6 = cross_validate(estimator=rfc, X = X, y = y.values.ravel(), scoring=scoring, cv = cv, return_train_score=True)
print("6")

f1 = [results1['test_f1'], results2['test_f1'], results3['test_f1'], results4['test_f1'], results5['test_f1'], results6['test_f1']]
precision = [results1['test_precision'], results2['test_precision'], results3['test_precision'], results4['test_precision'], results5['test_precision'], results6['test_precision']]
recall = [results1['test_recall'], results2['test_recall'], results3['test_recall'], results4['test_recall'], results5['test_recall'], results6['test_recall']]

df = pd.DataFrame(f1).transpose()
df.columns = ['biased', 'DT', 'NB', 'SVC', 'MLP', 'RF']
boxplot = df.boxplot(column=['biased', 'DT', 'NB', 'SVC', 'MLP', 'RF'])
plt.title('f1 score of 6 classifiers')
plt.show()
print("Mean of f1 of each classifier", df.mean())
print("Standard deviation of f1 of each classifier", df.std())


df = pd.DataFrame(precision).transpose()
df.columns = ['biased', 'DT', 'NB', 'SVC', 'MLP', 'RF']
boxplot = df.boxplot(column=['biased', 'DT', 'NB', 'SVC', 'MLP', 'RF'])
plt.title('precision score of 6 classifiers')
plt.show()
print("Mean of precision of each classifier", df.mean())
print("Standard deviation of precision of each classifier", df.std())

df = pd.DataFrame(recall).transpose()
df.columns = ['biased', 'DT', 'NB', 'SVC', 'MLP', 'RF']
boxplot = df.boxplot(column=['biased', 'DT', 'NB', 'SVC', 'MLP', 'RF'])
plt.title('recall score of 6 classifiers')
plt.show()
print("Mean of recall of each classifier", df.mean())
print("Standard deviation of recall of each classifier", df.std())


Classi = ['Biased Classifier', 'Decision Tree', 'GaussianNB', 'SVC', 'MLP Classifier', 'RandomForest Classifier']
w, p = wilcoxon(results2['test_f1'], results1['test_f1'])

for i in range(0, len(f1)):
    for j in range(i + 1, len(f1)):
        w, p = wilcoxon(f1[i], f1[j])
        print(f"Compare {Classi[i]} F1 to F1 of {Classi[j]}, p value =", p)
        if p <= 0.05:
            print("The difference is statistically significant.")
        else:
            print("The difference is not statistically significant.")

        w, p = wilcoxon(precision[i], precision[j])
        print(f"Compare {Classi[i]} precision to precision of {Classi[j]}, p value =", p)
        if p <= 0.05:
            print("The difference is statistically significant.")
        else:
            print("The difference is not statistically significant.")

        w, p = wilcoxon(recall[i], recall[j])
        print(f"Compare {Classi[i]} recall to recall of {Classi[j]}, p value =", p)
        if p <= 0.05:
            print("The difference is statistically significant.")
        else:
            print("The difference is not statistically significant.")

        print("------------------")






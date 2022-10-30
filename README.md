# Predicting bug 

This project predict bug proneness of classes from code and NLP metrics. 

•	In extract_feature_vectors.py, I extract feature vectors from java files such as number of block comments and statements, the result generated is in feature_vector_file.csv.
•	În label_feature_vectors.py, I labeled the buggy and non-buggy classes with 1 and 0. The labeled feature vectors are in new_feature_vector_file.csv
•	In train_classifiers.py, I apply classifiers (Decision Trees, Naive Bayes, Support Vector Machine, Multi-Layer Perceptron, and Random Forest) on the labeled feature vectors and tuning their hyperparameters using scikit-learn.
•	Finally, in evaluate_classifiers.py, I obtain precision, recall and F1 evaluation scores with 5-fold cross-validation technique, assessing the statistical significance with Wilcoxon test and presenting in boxplot.

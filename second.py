import csv
import pandas as pd
import matplotlib.pyplot as plt

import random
import numpy as np
import pandas as pd
from sklearn import datasets, svm, tree, preprocessing, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import sklearn.ensemble as ske
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



df = pd.read_csv('/Users/justinduan/Documents/MLProj1/inpatientCharges.csv')
df = df.drop(['Provider Id', 'Provider Name', 'Provider Street Address', 'Provider Zip Code', 'Hospital Referral Region Description'
              , ' Total Discharges ', ' Average Covered Charges ', 'Average Medicare Payments'], axis=1)
df = df.iloc[0:8000]





def preprocess_hospital_df(df):
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df['DRG Definition'] = le.fit_transform(processed_df['DRG Definition'])
    processed_df['Provider City'] = le.fit_transform(processed_df['Provider City'])
    processed_df['Provider State'] = le.fit_transform(processed_df['Provider State'])
    processed_df[' Average Total Payments '] = processed_df[' Average Total Payments '].map(lambda x: x.lstrip('$'))

    return processed_df

processed = preprocess_hospital_df(df)






x = processed.drop(['DRG Definition'], axis=1).values
y = processed['DRG Definition'].values
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

# DECISION TREE
# dt = open('titanicDT.csv', 'w')
# dt.write('depth' + ', ' + 'CV_SCORE' + ', ' + 'TRAIN_SCORE' + ', ' + 'TEST_SCORE\n')
# for depth in range(100):
#     model = tree.DecisionTreeClassifier(max_depth=(depth+1))
#     dt.write(str(depth+1) + ', ' + str(cross_val_score(model, X_train, y_train).mean()) + ', ')
#     model.fit(X_train, y_train)
#     dt.write(str(model.score(X_train, y_train)) + ', ')
#     dt.write(str(model.score(X_test, y_test)) + '\n')



# LEARNING CURVES, SUB OUT MODEL WITH THE DIFFERENT SUPERVISED LEARNING ALGOS TO COMPARE INPUT SIZE
# inputsize = preprocess_hospital_df(df)
# model = KNeighborsClassifier(n_neighbors=40)
# for i in range(16):
#     data = inputsize.iloc[0:((i+1)*500)]
#     x = processed.drop(['DRG Definition'], axis=1).values
#     y = processed['DRG Definition'].values
#     X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
#     print(str(cross_val_score(model, X_train, y_train, cv=10).mean()))
#
# print('stop')
#
# for i in range(16):
#     data = inputsize.iloc[0:((i+1)*500)]
#     x = processed.drop(['DRG Definition'], axis=1).values
#     y = processed['DRG Definition'].values
#     X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
#     model.fit(X_train, y_train)
#     print(str(model.score(X_train, y_train)))
#
# print('stop')
#
# for i in range(16):
#     data = inputsize.iloc[0:((i+1)*500)]
#     x = processed.drop(['DRG Definition'], axis=1).values
#     y = processed['DRG Definition'].values
#     X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2)
#     model.fit(X_train, y_train)
#     print(str(model.score(X_test, y_test)))








# ab = open('titanicAB.csv', 'w')
# ab.write('estimators' + ', ' + 'CV_SCORE' + ', ' + 'TRAIN_SCORE' + ', ' + 'TEST_SCORE\n')
# for estimator in range(100):
#     print(estimator)
#     boost = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=6), n_estimators=(estimator+1))
#     ab.write(str(estimator+1) + ', ' + str(cross_val_score(boost, X_train, y_train).mean()) + ', ')
#     boost.fit(X_train, y_train)
#     ab.write(str(boost.score(X_train, y_train)) + ', ')
#     ab.write(str(boost.score(X_test, y_test)) + '\n')



# LINEAR SVM
# linear = svm.LinearSVC()
# print(str(cross_val_score(linear, X_train, y_train, cv=10).mean()))
# linear.fit(X_train, y_train)
# print(linear.score(X_train, y_train))
# print(linear.score(X_test, y_test))
# # RBF SVM
# gaussian = svm.SVC(kernel='rbf')
# print(str(cross_val_score(gaussian, X_train, y_train, cv=10).mean()))
# gaussian.fit(X_train, y_train)
# print(gaussian.score(X_train, y_train))
# print(gaussian.score(X_test, y_test))
# # SIGMOID SVM
# sigmoid = svm.SVC(kernel='sigmoid')
# print(str(cross_val_score(sigmoid, X_train, y_train, cv=10).mean()))
# sigmoid.fit(X_train, y_train)
# print(sigmoid.score(X_train, y_train))
# print(sigmoid.score(X_test, y_test))
#


# KNN
# kn = open('titanicknn.csv', 'w')
# kn.write('neighbors' + ', ' + 'CV_SCORE' + ', ' + 'TRAIN_SCORE' + ', ' + 'TEST_SCORE\n')
# for neighbor in range(100):
#     print(neighbor)
#     knn = KNeighborsClassifier(n_neighbors=neighbor+1)
#     kn.write(str(neighbor + 1) + ', ' + str(cross_val_score(knn, X_train, y_train, cv=10).mean()) + ', ')
#     knn.fit(X_train, y_train)
#     kn.write(str(knn.score(X_train, y_train)) + ', ')
#     kn.write(str(knn.score(X_test, y_test)) + '\n')


# NEURAL NET RELU
# nn = open('titanicnn.csv', 'w')
# nn.write('neurons' + ', ' + 'CV_SCORE' + ', ' + 'TRAIN_SCORE' + ', ' + 'TEST_SCORE\n')
# for neuron in range(100):
#     print(neuron)
#     neural = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(neuron+1))
#     nn.write(str(neuron + 1) + ', ' + str(cross_val_score(neural, X_train, y_train, cv=10).mean()) + ', ')
#     neural.fit(X_train, y_train)
#     nn.write(str(neural.score(X_train, y_train)) + ', ')
#     nn.write(str(neural.score(X_test, y_test)) + '\n')

# NEURAL NET TANH
# nn = open('titanicnn.csv', 'w')
# nn.write('neurons' + ', ' + 'CV_SCORE' + ', ' + 'TRAIN_SCORE' + ', ' + 'TEST_SCORE\n')
# for neuron in range(100):
#     print(neuron)
#     neural = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(neuron+1), activation='tanh')
#     nn.write(str(neuron + 1) + ', ' + str(cross_val_score(neural, X_train, y_train, cv=10).mean()) + ', ')
#     neural.fit(X_train, y_train)
#     nn.write(str(neural.score(X_train, y_train)) + ', ')
#     nn.write(str(neural.score(X_test, y_test)) + '\n')











import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC

file = 'data.csv'
dir = os.getcwd()
file = os.path.join(dir, file)

df = pd.read_csv(file) # read csv data in df
# print(df)

# filename chroma_stft spectral_centroid spectral_bandwidth rolloff zero_crossing_rate mfcc1 ... mfcc20 label

# x = df[['chroma_stft', 'spectral_centroid']].values
# x = df[['chroma_stft', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate', 'mfcc1',
#         'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',
#         'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']].values

x = df[['chroma_stft', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate', 'mfcc1',
         'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',
         'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']].values

# scaler = StandardScaler()
# scaler.fit(x)
# x = scaler.transform(x)

y = df['label'].values # blues, classical, country, ...
labels = y
le = LabelEncoder()
le.fit(y)
y_encoded = le.transform(y) # transform labels from strings to numeric values

test_split = 0.3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, random_state=4)

# linear svm
c_list = [1, 10]
print('linear svm')
for c in c_list:
    linear_svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=c, loss="hinge"))
    ))
    linear_svm_clf.fit(x_train, y_train)
    # plot_decision_boundary(linear_svm_clf, x_train, y_train, c)
    y_pred = linear_svm_clf.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print('c: ' + str(c) + ' accuracy: ' + str(acc))

#
c_list = [1, 10]
print('poly svm')
for c in c_list:
    polynomial_svm_clf = Pipeline((
        ("poly_features", PolynomialFeatures(degree=2)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=c, loss="hinge"))
        ))
    polynomial_svm_clf.fit(x_train, y_train)
    y_pred = polynomial_svm_clf.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print('c: ' + str(c) + ' accuracy: ' + str(acc))
    # plot_decision_boundary(polynomial_svm_clf, x_train, y_train, c)

# poly kernel svm
c_list = [1, 10]
degree_list = [2, 3]
print('poly kernel svm')
for c in c_list:
    for degree in degree_list:
        poly_kernel_svm_clf = Pipeline((
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="poly", degree=degree, coef0=1, C=c))
        ))
        poly_kernel_svm_clf.fit(x_train, y_train)
        y_pred = poly_kernel_svm_clf.predict(x_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        metrics.plot_confusion_matrix(poly_kernel_svm_clf, x_test, y_test)

        print('degree ' + str(degree) + 'poly kernel svm c: ' + str(c) + ' accuracy: ' + str(acc))
        # plot_decision_boundary(poly_kernel_svm_clf, x_train, y_train)

# kernel_list = ['rbf', 'poly', 'sigmoid']
kernel_list = ['rbf']
c_list = [1, 10]
gamma_list = [0.01, 0.1]
print('rbf kernel svm')
for kernel in kernel_list:

    for c in c_list:
        for gamma in gamma_list:
            rbf_kernel_svm_clf = Pipeline((
                ("scaler", StandardScaler()),
                ("svm_clf", SVC(kernel=kernel, gamma=gamma, C=c))
            ))
            rbf_kernel_svm_clf.fit(x_train, y_train)
            # plot_decision_boundary(rbf_kernel_svm_clf, x_train, y_train)
            y_pred = rbf_kernel_svm_clf.predict(x_test)
            acc = metrics.accuracy_score(y_test, y_pred)
            print('kernel ' + kernel + ' gamma: ' + str(gamma) + ' c: ' + str(c) + ' accuracy: ' + str(acc))

plt.show()





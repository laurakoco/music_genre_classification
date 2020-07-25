
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

file = 'data.csv'
dir = os.getcwd()
file = os.path.join(dir, file)

def plot_performance(df):

    plt.figure()

    x = df['model'].values
    y = df['acc'].values

    plt.bar(x,y)
    plt.xticks(rotation=45)
    plt.title('Mean Classifier Accuracy')
    plt.tight_layout()
    plt.grid()

def get_model_performance(df,model,model_name):

    acc_list = []
    precision_list = []
    recall_list = []

    num = 10

    for i in range(0, num):

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc_list.append(metrics.accuracy_score(y_test, y_pred))
        precision_recall_fscore_support = metrics.precision_recall_fscore_support(y_test, y_pred)
        precision_list.append(np.mean(precision_recall_fscore_support[0]))
        recall_list.append(np.mean(precision_recall_fscore_support[1]))

    acc = np.mean(acc_list)
    precision = np.mean(precision_list)
    recall = np.mean(recall_list)

    df_model = pd.DataFrame([[model_name, acc, precision, recall]], columns=['model','acc','precision','recall'])
    df = df.append(df_model)

    return df

if __name__ == "__main__":

    df = pd.read_csv(file) # read csv data in df

    x = df[['chroma_stft', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate', 'mfcc1',
             'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',
             'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']].values

    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    y = df['label'].values # blues, classical, country, ...

    test_split = 0.1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, random_state=4)

    df = pd.DataFrame(columns=['model','acc','precision','recall']) # df for storing error for d and N combinations

    # linear svm
    model = linear_svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge", max_iter=1000000))
    ))
    df = get_model_performance(df,model,'Linear SVM')

    # poly kernel svm
    model = poly_kernel_svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=2, coef0=1, C=10))
    ))
    df = get_model_performance(df,model,'Poly Kernel SVM')

    # rbf svm
    model = rbf_kernel_svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel='rbf', gamma=0.01, C=10))
    ))
    df = get_model_performance(df,model,'RBF SVM')

    # k-nn
    k = 7
    model = KNeighborsClassifier(n_neighbors=k)
    model_name = 'k-NN k=' + str(k)
    df = get_model_performance(df,model,model_name)

    # logistic regression
    model = LogisticRegression(max_iter=10000).fit(x_train, y_train)
    df = get_model_performance(df,model,'Logistic Regression')

    # naive bayesian
    model = GaussianNB()
    df = get_model_performance(df,model,'Naive Bayesian')

    # lda
    model = LDA()
    df = get_model_performance(df,model,'LDA')

    # qda
    model = QDA()
    df = get_model_performance(df,model,'QDA')

    # random forest
    model = RandomForestClassifier(n_estimators=6, max_depth=10, criterion='entropy')
    df = get_model_performance(df,model,'Random Forest')

    # adaboost

    # decision tree
    model = tree.DecisionTreeClassifier(criterion='entropy')
    df = get_model_performance(df,model,'Decision Tree')

    print(df)

    df.to_csv('model_performance.csv',index=False) # write to csv

    plot_performance(df)

    plt.show()

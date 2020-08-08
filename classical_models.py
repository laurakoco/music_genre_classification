
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
from sklearn.ensemble import VotingClassifier

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

    metrics.plot_confusion_matrix(model, x_test, y_test)
    plt.title(model_name)

    df_model = pd.DataFrame([[model_name, acc, precision, recall]], columns=['model','acc','precision','recall'])
    df = df.append(df_model)

    return df

if __name__ == "__main__":

    df = pd.read_csv(file) # read csv data in df

    x = df[['chroma_stft', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate', 'mfcc1',
             'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',
             'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']].values

    y = df['label'].values # blues, classical, country, ...

    test_split = 0.1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, random_state=4)

    # fit scaling on training data only
    # then apply to test data
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)

    x_test = scaler.transform(x_test)

    df = pd.DataFrame(columns=['model','acc','precision','recall']) # df for storing error for d and N combinations

    # linear svm
    lin_svm = linear_svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge", max_iter=1000000))
    ))
    df = get_model_performance(df,lin_svm,'Linear SVM')

    # poly kernel svm
    poly_svm = poly_kernel_svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=2, coef0=1, C=10))
    ))
    df = get_model_performance(df,poly_svm,'Poly Kernel SVM')

    # rbf svm
    rbf_svm = rbf_kernel_svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel='rbf', gamma=0.1, C=10))
    ))
    df = get_model_performance(df,rbf_svm,'RBF SVM')

    # k-nn
    k = 7
    knn = KNeighborsClassifier(n_neighbors=k)
    model_name = 'k-NN k=' + str(k)
    df = get_model_performance(df,knn,model_name)

    # logistic regression
    lr = LogisticRegression(max_iter=10000).fit(x_train, y_train)
    df = get_model_performance(df,lr,'Logistic Regression')

    # naive bayesian
    nb = GaussianNB()
    df = get_model_performance(df,nb,'Naive Bayesian')

    # lda
    lda = LDA()
    df = get_model_performance(df,lda,'LDA')

    # qda
    qda = QDA()
    df = get_model_performance(df,qda,'QDA')

    # random forest
    rf = RandomForestClassifier(n_estimators=6, max_depth=10, criterion='entropy')
    df = get_model_performance(df,rf,'Random Forest')

    # adaboost

    # decision tree
    dt = tree.DecisionTreeClassifier(criterion='entropy')
    df = get_model_performance(df,dt,'Decision Tree')

    # ensemble (majority) voting classifier
    vc = VotingClassifier(
        estimators = [('poly_svm',poly_svm),('qda',qda),('rbf_svm',rbf_svm),('knn',knn)],
        voting = 'hard')
    df = get_model_performance(df,vc,'Voting Classifier')

    # e

    print(df)

    df.to_csv('model_performance.csv',index=False) # write to csv

    plot_performance(df)

    plt.show()

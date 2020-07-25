
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

def get_model_performance(model):

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

    print('accuracy: ' + str(round(acc,4)))
    # print('precision: ' + str(round(acc,4)))
    # print('recall: ' + str(round(acc,4)))

    # metrics.plot_confusion_matrix(poly_kernel_svm_clf, x_test, y_test)

    return acc

def plot_performance_poly(df):

    plt.figure()

    degree = df.shape[0]
    col_list = list(df.columns)

    x = df['degree'].values

    for col in col_list:
        if col == 'degree':
            pass
        else:
            y = df[col].values
            plt.plot(x, y, label='c=' + col)

    plt.grid()
    plt.legend()

    plt.title('Poly Kernel SVM')
    plt.xlabel('Degree')
    plt.ylabel('Mean Accuracy')

def plot_performance_rbf(df):

    plt.figure()

    col_list = list(df.columns)

    x = df['gamma'].values

    for col in col_list:
        if col == 'gamma':
            pass
        else:
            y = df[col].values
            plt.semilogx(x, y, label='kernel=' + col)

    plt.grid()
    plt.legend()

    plt.title('SVM with Different Kernels')
    plt.xlabel('Gamma')
    plt.ylabel('Mean Accuracy')



if __name__ == "__main__":


    file = 'data.csv'
    dir = os.getcwd()
    file = os.path.join(dir, file)

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

    # linear svm
    print('linear svm')
    c_list = [1]
    for c in c_list:
            print('c:' + str(c))
            linear_svm_clf = Pipeline((
                ("scaler", StandardScaler()),
                ("linear_svc", LinearSVC(C=c, loss="hinge", max_iter=100000))
            ))
            acc = get_model_performance(linear_svm_clf)

    # poly kernel svm
    print('poly kernel svm')
    c_list = [1,10]
    degree_list = [1,2,3,4,5]

    df_poly = pd.DataFrame(columns=['degree','1','10']) # df for storing error for c and degree combinations

    df_poly['degree'] = degree_list

    for c in c_list:
        acc_list = []
        for degree in degree_list:
            print('c:' + str(c))
            print('degree:' + str(degree))
            poly_kernel_svm_clf = Pipeline((
                ("scaler", StandardScaler()),
                ("svm_clf", SVC(kernel="poly", degree=degree, coef0=1, C=c))
            ))
            acc_list.append(get_model_performance(poly_kernel_svm_clf))
        df_poly[str(c)] = acc_list

    plot_performance_poly(df_poly)

    print(df_poly)

    # rbf kernel svm
    print('rbf kernel svm')
    kernel_list = ['rbf','poly','sigmoid']
    c_list = [10]
    gamma_list = [0.001,0.01,0.1,1]

    df_rbf = pd.DataFrame(columns=['gamma','rbf','poly','sigmoid']) # df for storing error for c and degree combinations

    df_rbf['gamma'] = gamma_list

    for c in c_list:
        for kernel in kernel_list:
            acc_list = []
            for gamma in gamma_list:
                print('c:' + str(c))
                print('kernel:' + kernel)
                print('gamma:' + str(gamma))
                rbf_kernel_svm_clf = Pipeline((
                        ("scaler", StandardScaler()),
                        ("svm_clf", SVC(kernel=kernel, gamma=gamma, C=c))
                ))
                acc_list.append(get_model_performance(rbf_kernel_svm_clf))
            df_rbf[kernel] = acc_list

    print(df_rbf)

    plot_performance_rbf(df_rbf)

    plt.show()





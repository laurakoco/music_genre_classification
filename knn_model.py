
import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

file = 'data.csv'
dir = os.getcwd()
file = os.path.join(dir, file)

df = pd.read_csv(file) # read csv data in df

x = df[['chroma_stft', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate', 'mfcc1',
         'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',
         'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']].values

y = df['label'].values  # blues, classical, country, ...

test_split = 0.1
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, random_state=4)

# fit scaling on training data only
# then apply to test data
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)

num = 10
k_list = [3,5,7,9,11,13,15,17,19,21,23,25] # find optimal k
acc_list = []
for k in k_list:
    acc = []
    precision = []
    recall = []
    for i in range(0,num):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train,y_train)
        y_pred = knn.predict(x_test)
        acc.append(metrics.accuracy_score(y_test, y_pred))
        # precision_recall_fscore_support = metrics.precision_recall_fscore_support(y_test, y_pred)
        # precision.append(np.mean(precision_recall_fscore_support[0]))
        # recall.append(np.mean(precision_recall_fscore_support[1]))
    acc_list.append(np.mean(acc))
    # print('k=' + str(k))
    # print('acc: ' + str(np.mean(acc)))
    # print('precision: ' + str(np.mean(precision)))
    # print('recall: ' + str(np.mean(recall)))

plt.scatter(k_list,acc_list)
plt.grid()
plt.xlabel('k')
plt.ylabel('Mean Accuracy')
plt.title('k-NN Accuracy')

plt.show()
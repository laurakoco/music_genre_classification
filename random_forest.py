
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def plot_acc(df):

    plt.figure()

    num_row = df.shape[0]
    num_col = df.shape[1]

    x = np.arange(1,num_row+1)

    for i in range(1,num_col+1):
        y = df[str(i)].values
        plt.plot(x, y, label='d='+str(i))

    plt.grid()
    plt.legend()

    plt.title('Random Forest Accuracy')
    plt.xlabel('N')
    plt.ylabel('Mean Accuracy')

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

df = pd.DataFrame(columns=['1','2','3','4','5'])  # df for storing error for d and N combinations

N_list = np.arange(10,20)
d_list = np.arange(1,11)

num = 5

for d in d_list:
    acc_list = []
    for N in N_list:
        acc = []
        for i in range(0,num):
            model = RandomForestClassifier(n_estimators=N,max_depth=d,criterion='entropy')
            model = model.fit(x_train,y_train)
            y_pred = model.predict(x_test)
            acc.append(metrics.accuracy_score(y_test,y_pred))
        acc_list.append(np.mean(acc))
    df[str(d)] = acc_list

plot_acc(df)

print(df)

plt.show()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

def plot_decision_boundary(model, x, y):

    plt.figure()

    # plot decision regions
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    plt.contourf(xx, yy, z, alpha=0.4)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=20, edgecolor='k')
    # plt.legend()

file = 'data.csv'
dir = os.getcwd()
file = os.path.join(dir, file)

df = pd.read_csv(file) # read csv data in df

x = df[['spectral_centroid', 'spectral_bandwidth']].values
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

y = df['label'].values # blues, classical, country, ...
labels = df['label'].unique()

le = LabelEncoder()
le.fit(y)
y = le.transform(y) # transform labels from strings to numeric values

test_split = 0.3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, random_state=4)

# linear svm
c = 1
linear_svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=c, loss="hinge"))
))
# linear_svm_clf.fit(x_train, y_train)
# plot_decision_boundary(linear_svm_clf, x_train, y_train)
# y_pred = linear_svm_clf.predict(x_test)
# acc = metrics.accuracy_score(y_test, y_pred)
# print('c: ' + str(c) + ' accuracy: ' + str(acc))

# plt.scatter(x[:,0], x[:,1], c=y, label=labels)
# plt.grid()


fig, ax = plt.subplots()
for label in labels:
    df_label = df[df['label'] == label]
    x = df_label[['zero_crossing_rate', 'chroma_stft']].values
    scaler.fit(x)
    x = scaler.transform(x)
    y = df_label['label'].values
    ax.scatter(x[:,0], x[:,1], label=label)

plt.legend()
plt.grid()
plt.xlabel('Zero Crossing Rate')
plt.ylabel('Chroma STFT')

fig = plt.figure()
ax = Axes3D(fig)

for label in labels:
    df_label = df[df['label'] == label]
    x = df_label[['zero_crossing_rate', 'chroma_stft', 'spectral_centroid']].values
    scaler.fit(x)
    x = scaler.transform(x)
    y = df_label['label'].values
    ax.scatter(x[:,0], x[:,1], x[:,2], label=label)

plt.legend()
plt.grid()
ax.set_xlabel('Zero Crossing Rate')
ax.set_ylabel('Chroma STFT')
ax.set_zlabel('Spectral Centroid')







plt.show()

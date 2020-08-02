
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import metrics

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
labels = y
le = LabelEncoder()
le.fit(y)
y = le.transform(y)

test_split = 0.1
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, random_state=4)

input_shape = x_train.shape[1]

num_classes = len(y_train)
# reset underlying graph data
# tf.reset_default_graph()

# build nn model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(256, activation='relu', input_shape=(x_train.shape[1],)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

tf.keras.callbacks.History()

# train model
model.fit(x_train,
          y_train,
          epochs = 10,
          batch_size = 128,
          validation_split = 0.2)

test_loss, test_acc = model.evaluate(x_test, y_test)

pred = model.predict_classes(x_test)
acc = metrics.accuracy_score(y_test, pred)
precision_recall_fscore_support = metrics.precision_recall_fscore_support(y_test, pred)
prec = np.mean(precision_recall_fscore_support[0])
recall = np.mean(precision_recall_fscore_support[1])

print('accuracy: ' + str(acc))
print('precision: ' + str(prec))
print('recall: ' + str(recall))

print(model.history.history)

epoch = model.history
epoch = model.history.epoch
acc = model.history.history['accuracy']
loss = model.history.history['loss']
val_loss = model.history.history['val_loss']
val_acc = model.history.history['val_accuracy']

# print('test_acc: ',test_acc)

fig, ax1 = plt.subplots()

plt.title('Training Data')

ax2 = ax1.twinx()

line1 = ax1.plot(epoch,acc,label='accuracy')
line2 = ax1.plot(epoch,val_acc,label='val_accuracy')

line3 = ax2.plot(epoch,loss,label='loss',color='g')
line4 = ax2.plot(epoch,val_loss,label='val_loss',color='r')

lines = line1+line2+line3+line4
labs = [l.get_label() for l in lines]
ax1.legend(lines, labs, loc=1)

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax2.set_ylabel('Loss')

plt.grid()

plt.show()

# load model

result = model.predict(x_test)
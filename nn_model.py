

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from tensorflow.python.compiler.tensorrt import trt_convert as trt

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC

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

test_split = 0.3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, random_state=4)


input_shape = x_train.shape[1]

num_classes = len(y_train)
# reset underlying graph data
# tf.reset_default_graph()

# build nn model
model = tf.keras.Sequential()
# model.add(tf.keras.Input(shape=input_shape,))
# model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

model.add(tf.keras.layers.Dense(256, activation='relu', input_shape=(x_train.shape[1],)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# train model
model.fit(x_train, y_train, epochs = 50, batch_size = 128)

test_loss, test_acc = model.evaluate(x_test, y_test)

print('test_acc: ',test_acc)


# pickle.dump( {'my_words':my_words, 'my_classes


# load model

result = model.predict(x_test)
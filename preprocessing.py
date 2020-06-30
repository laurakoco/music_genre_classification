
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

map = plt.get_cmap('inferno')

path = '/Users/laura/Documents/Data Science/Project/genres/' # path to data

# audio_path = '/Users/laura/Documents/Data Science/Project/genres/metal/metal.00000.wav'

#img_path = '/Users/laura/Documents/Data Science/Project/img/'

#cmap = plt.get_cmap('inferno')

# plt.figure()
# # genre_list = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
# genre_list = ['blues']
# for g in genre_list:
#     for filename in os.listdir(path + g):
#         file_path = path + g + '/' + filename
#         y, sr = librosa.load(file_path, mono=True, duration=5)
#         plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
#         plt.axis('off')
#         plt.savefig(img_path + filename + '.png')
#         plt.clf()

header = 'filename chroma_stft spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

file = open('data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

genre_list = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
for genre in genre_list:
    for filename in os.listdir(path + genre):
        file_path = path + genre + '/' + filename
        y, sr = librosa.load(file_path, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {genre}'
        file = open('data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
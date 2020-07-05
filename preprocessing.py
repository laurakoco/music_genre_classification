
import librosa
import numpy as np
import pandas as pd
import os

path = '/Users/laura/Documents/Data Science/Project/genres/' # path to data

header = ['filename', 'chroma_stft', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate']

# add mfc coefficients to header: mfcc1, mfcc2, ... mfcc20
for i in range(1, 21):
    header.append('mfcc' + str(i))

header.append('genre')

df = pd.DataFrame (columns = header)

genre_list = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
row_num = 0
for genre in genre_list: # iterate through genres
    for filename in os.listdir(path + genre): # iterate through files in genre
        print(filename)
        file_path = path + genre + '/' + filename
        y, sr = librosa.load(file_path, mono=True, duration=30) # load wav file
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        # create list of data to append to row - take mean of each feature
        data = [filename, np.mean(chroma_stft), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)]
        for i in mfcc:
            data.append(np.mean(i))
        data.append(genre)
        df.loc[row_num] = data # add row
        row_num += 1

df.to_csv('data.csv',index=False) # write df to csv
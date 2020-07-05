# Music Genre Classification in Python

Analyze and classify audio signals in Python

<img src="images/3d_plot.png" width="600">

## Built With

* Python 3.7
* [Librosa](https://librosa.org/librosa/) 0.7.2
* [sklearn](https://scikit-learn.org/stable/) 0.0
* [TensorFlow](https://www.tensorflow.org/) 2.1.1
* [pandas](https://pandas.pydata.org/) 1.0.4

## Data

[GTZAN](http://marsyas.info/downloads/datasets.html) Genre Collection

* Dataset consists of 1000 audio tracks, 30 seconds long each
* Tracks are all 22050 Hz mono 16-bit audio files in .wav format
* 10 genres (100 songs/genre)
	* Blues
	* Classical
	* Country
	* Disco
	* Hip Hop
	* Jazz
	* Metal
	* Pop
	* Raggae
	* Rock

### Feature Extraction

* Zero crossing rate - rate at which the signal changes from positive to negative or negative to positive
* Spectral centroid - weighted mean of frequencies present in audio clip
* Spectral roll-off - the frequency below which a specified percentage of the total spectral energy ( 85% by default) lies
* Chroma frequencies - the intensity of each of the 12 distinct musical chroma of the octave; chroma representation/chromagarm (via short-term fourier transform) 
* Mel-Frequency Cepstral Coefficients (MFCC) (x20) - coefficients that collectively make up an MFC
	* Mel-Frequency Cepstrum (MFC) - representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency

## Results

<img src="images/confusion_matrix_svm_poly_kernel.png" width="600">


|                    | Mean Accuracy | Mean Precision | Mean Recall |
|:------------------:|:-------------:|:--------------:|:-----------:|
| SVM, Linear Kernel |      0.61     |      0.60      |     0.61    |
|  SVM, Poly Kernel  |      0.77     |      0.78      |     0.77    |
|   SVM, RBF Kernel  |      0.76     |      0.76      |     0.76    |
|         NN         |      0.67     |                |             |


## Usage

* Download GTZAN dataset
* Run requirements.txt
```
$ pip install -r requirements.txt
```
* Run preprocessing.py to generate csv file (data.csv) with features for each file
	* Change path to the root directory (genres) of GTZAN dataset
```
path = '/path/to/gtzan/genres/' # path to data
```
* Run models (i.e. svm_model.py)


## Author

**Laura Kocubinski** [laurakoco](https://github.com/laurakoco)

## Acknowledgments

* Boston University MET Master of Science Computer Science Program
* MET CS 677 Data Science with Python

## References

[1] https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8 

[2] "Musical Genre Classification of Audio Signals" G. Tzanetakis, P. Cook. IEEE Transactions on Audio and Speech Processing, 2002.

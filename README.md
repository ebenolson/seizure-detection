# Kaggle UPenn and Mayo Clinic's Seizure Detection Challenge

This repository contains documentation and code for the second place submission by Eben Olson and Damian Mingle.

http://www.kaggle.com/c/seizure-detection

##Hardware / OS platform used

 * EC2 m3.2xlarge instances (8 vCPU, 30GB RAM)
 * Core i5-2400 quad core @ 3.10GHz, 16GB RAM
 * Ubuntu linux 12.04

##Dependencies

 * Python 2.7
 * IPython 2.1.0
 * Theano 0.6.0
 * scikit_learn-0.14.1
 * numpy-1.8.1
 * scipy-0.14.0
 * joblib-0.8.3-r1 (only needed for multi-core batch processing)

##Steps to train the model and obtain a submission

* Obtain the competition data and uncompress it into the 'data/clips' directory of the project.
```
data/
  clips/
    Dog_1/
      Dog_1_ictal_segment_1.mat
      Dog_1_ictal_segment_2.mat
      ...
      Dog_1_interictal_segment_1.mat
      Dog_1_interictal_segment_2.mat
      ...
      Dog_1_test_segment_1.mat
      Dog_1_test_segment_2.mat
      ...

    Dog_2/
    ...
```

* Run IPython from inside the project directory with pylab mode enabled
```
ipython --pylab
```

* Open the notebook "Data Preprocessing.ipnyb" in your browser and execute all cells to prepare data for training.

* Open the notebook "Train Classifiers and Predict.ipynb" and execute all cells to train models and save predictions to the "output" directory.

* Open the notebook "Postprocessing.ipynb" and execute all cells to average the model predictions and produce "submission.csv" in the project root directory.

## View notebooks online at NBViewer
* [Data Preprocessing](http://nbviewer.ipython.org/github/ebenolson/seizure-detection/blob/master/Data%20Preparation.ipynb)
* [Train Classifiers and Predict](http://nbviewer.ipython.org/github/ebenolson/seizure-detection/blob/master/Train%20Classifiers%20and%20Predict.ipynb)
* [Postprocessing](http://nbviewer.ipython.org/github/ebenolson/seizure-detection/blob/master/Postprocessing.ipynb)

## Project report

Available at https://github.com/ebenolson/seizure-detection/raw/master/report.pdf
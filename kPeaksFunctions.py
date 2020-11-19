# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:42:10 2020

@author: Harry Bowman
"""

import numpy as np
import csv
import os
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import find_peaks
from sklearn.svm import OneClassSVM
#import scipy.fft
import pandas as pd

#%%FFT Function
def getFFT(sample, axisLen = 2048):
        x_fft_ = fft(sample, axisLen)
        x_fft = 2.0 / math.sqrt(axisLen) * np.abs(x_fft_[0:axisLen//2])
        return x_fft

#%% smoothing function
def movingAvCust(x, w = 6, ss = 1):
    smoothedX = []
    for i in np.arange(w/2, len(x)-w/2, ss):
        val = np.mean(x[int(i-w/2):int(i+w/2)])
        smoothedX.append(val)
    return np.array(smoothedX)

#%%k peaks finder
def kPeaks(wave, numPeaks = 3, width = 20, minProminence = 1, minHeightDivider = 5):
    
    #get peaks
    peaks, prop = find_peaks(wave, distance=width, prominence = (minProminence, None))
    if len(peaks) != 0:
        prom = prop['prominences']
        
        #Get max height and threshold out low peaks
        maxHeight= np.max(wave[peaks])
    #    print(maxHeight)
        minNormHeight = maxHeight/minHeightDivider
        goodPeaks = np.where(wave[peaks]> minNormHeight)
        peaks = peaks[goodPeaks]
        prom = prom[goodPeaks]
        
        
        #Find top 3 from this based on prominence
    #    topPeaks = np.sort(peaks[np.argsort(prom)[-numPeaks:]])
    else:
        peaks = []
        prom = []
        
    return peaks, prom

#%% GMM Function - put into sep file later

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import math
import matplotlib as mpl

COLOR = 'black'
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR

### ------------------ PYTHON FUNCTIONS TO IMPLEMENT CLUSTERING -----------------------------------
from sklearn.neighbors import LocalOutlierFactor

def lofClustering(data,plot = False, **kwargs):
    # Run LOF on the mean and std data for the trains
    lof = LocalOutlierFactor(n_neighbors=8, contamination='auto')
    lof.fit_predict(data)
    
    # Get the negative outlier factor for the trains
    nof = lof.negative_outlier_factor_
    
    # Run thresholding on the LOF to identify anomalous trains
    labels = (nof < -2.5).astype(int)
    
    if plot:
        labelColours = 'white'

        f1 = plt.figure(1)
        plt.title('Negative outlier factor value for cutoff', color='white')
        plt.plot(-nof,'bo', alpha=0.3)
    #     plt.show()
        f2 = plt.figure(figsize=(10, 6))

        ax = f2.add_subplot(111)
        colours = np.array(['blue','red'])
        ax.scatter(data[:, 0], data[:, 1], s=20, color=colours[labels], alpha=0.3)
        ax.set_title('Local outlier factor anomalies (adjusted cutoff = 2.5)', color='white')

        ax.set_xlabel('Mean normalised')#, fontsize = 15.0)
        ax.set_ylabel('Standard deviation normalised')

        plt.show()
    
    return labels

# ----------------------- Gaussian mixture models ----------------------------------------
from sklearn.mixture import GaussianMixture
import itertools
from scipy import linalg
import time

def gmmClustering(data, plot = False,numComponents = 7, **kwargs):
    st = time.time()

    lowest_aic = np.infty
    best_mod = ''
    best_num = 0

    aic = []
    n_components_range = range(1, numComponents)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:

            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(data)
            aicCurr = gmm.aic(data)
            aic.append(aicCurr)
            # print(f)

            # print(f'num k = {n_components}\t cv_type = {cv_type}\t curr score = {gmm.score(data)}')
            if aicCurr < lowest_aic:
                lowest_aic = aicCurr
                best_gmm = gmm
                best_mod = cv_type
                best_num = n_components

    aic = np.array(aic)

    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange'])
    clf = best_gmm

    Y_ = clf.predict(data)
    yUniq, yCount = np.unique(Y_, return_counts=True)
    # print(yUniq, yCount)

    p = 5 # Percentage of trains to consider a small cluster
    smallClusters = [val for i, val in enumerate(yUniq) if yCount[i] < (p/100 * sum(yCount))]

    labels = [1 if val in smallClusters else 0 for val in Y_]
#   anomalies = [trains[i] for i, val in enumerate(labels) if val==1]
#   # print(anomalies)
##  
#
#   anomalyDates = []
#   if kwargs.get('dates', None) is not None:
#       dates = kwargs.get('dates')
#
#       anomalyDates = [dates[i] for i, val in enumerate(labels) if val==1]

    ## Plotting of the various covariance matrix types
    if plot:
        bars = []

        # Plot the BIC scores
        plt.figure(figsize=(12, 15))
        spl = plt.subplot(2, 1, 1)
        for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
            xpos = np.array(n_components_range) + .2 * (i - 2)
            bars.append(plt.bar(xpos, aic[i * len(n_components_range):
                                          (i + 1) * len(n_components_range)],
                                width=.2, color=color))
        plt.xticks(n_components_range)
        plt.ylim([aic.min() * 1.01 - .01 * aic.max(), aic.max()])
        plt.title('AIC score per model')
        xpos = np.mod(aic.argmin(), len(n_components_range)) + .65 +\
            .2 * np.floor(aic.argmin() / len(n_components_range))
        plt.text(xpos, aic.min() * 0.97 + .03 * aic.max(), '*', fontsize=14)
        spl.set_xlabel('Number of components')
        legend = spl.legend([b[0] for b in bars], cv_types)
        plt.setp(legend.get_texts(), color='black')

        # Plot the winner
        splot = plt.subplot(2, 1, 2)
        

        for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                                   color_iter)):
            v, w = linalg.eigh(cov)
            if not np.any(Y_ == i):
                continue
            plt.scatter(data[Y_ == i, 0], data[Y_ == i, 1], color=color, alpha=0.5)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan2(w[0][1], w[0][0])
            angle = 180. * angle / np.pi  # convert to degrees
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            ell = Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(.5)
            splot.add_artist(ell)

#        plt.xticks(())
#        plt.yticks(())
        plt.title(f'Selected GMM: {best_mod}, {best_num} components')
    plt.xlabel('Magnitude')
    plt.ylabel('Frequency')
    plt.show()

    # print('\nGMM took %0.3f\n' % (time.time() - st))
    return labels



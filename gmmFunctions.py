# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:11:13 2020

@author: Harry Bowman
"""
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import MinMaxScaler as dataScaler
from sklearn.mixture import GaussianMixture as GM
from sklearn.metrics import silhouette_score as silScore
from sklearn.metrics import davies_bouldin_score as dbScore
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
from scipy.special import logsumexp


def gmmFit(data, numMixRange = 10, scoreToUse = 'sil', threshold=0.9):
    scaler = dataScaler().fit(data)
    X = scaler.transform(data)
    
    scoreList = []
    #scoreToUse = 'db' #or sil, or nothing for my old bad way 
    if scoreToUse == 'db':
        bestScore = -10000
    else:
        bestScore= 10000
    for i in np.arange(2,numMixRange+1):
            
        gmm = GM(i).fit(X)
        labels = gmm.predict(X)
        
        
        if scoreToUse == 'sil':
            currentScore = silScore(X, labels)
            if currentScore < bestScore:
                bestScore = currentScore
                bestGmm  = gmm
                bestMix = i
                
        elif scoreToUse == 'db':
            currentScore = dbScore(X, labels)
            if currentScore > bestScore:
                bestScore = currentScore
                bestGmm  = gmm
                bestMix = i
        
        else:
            currentScore = gmm.bic(X) + (20*(i**1.5/numMix))
            if currentScore > bestScore:
                bestScore = currentScore
                bestGmm  = gmm
                bestMix = i
        
        scoreList.append(currentScore)
    
    clf = bestGmm
    return clf, bestMix, scaler, bestScore
  
    
    
    
    
def gmmPredict(data, clf, scaler, likelihood_threshold, titleVal = 'PROVIDE TITLE VAL', bestMix = 88, bestScore = 0, plot = False):
    ###############################
    ###PREDICT ON THE DATA
    ###############################
    
    X = scaler.transform(data)
    score_samples = clf.score_samples(X)
    prob = -clf.score_samples(X) + likelihood_threshold
    numOutlier = len(prob[prob>likelihood_threshold])
    percentOutlier = numOutlier/len(data) * 100
    # percentOutlier = np.mean(prob,0)
    
    if plot:
        ######################################
        #PREDICT ON THE MESH
        ######################################
        #Generate background for plotting against
        freq = X[:,0]
        mag = X[:,1]
        print(mag.min())
        print(mag.max())
        meshSize = 200
        plt.figure()
        ax1 = plt.subplot(1,2,1)
        plt.scatter(freq,mag,c=prob>likelihood_threshold,s=8)
        xx, yy = np.meshgrid(np.linspace(freq.min(), freq.max(), meshSize), np.linspace(mag.min(), mag.max(), meshSize))
        U = np.concatenate([xx[...,None],yy[...,None]],2).reshape([-1,2])

        
        #get mesh probabilities
        score_samplesMesh = clf.score_samples(U)
        prob_U = -clf.score_samples(U)+likelihood_threshold
        ax2= plt.subplot(1,2,2, sharex = ax1, sharey = ax1)
        plt.scatter(U[:,0],U[:,1],c=prob_U>likelihood_threshold,s=8)
        uPlot = prob_U.reshape([meshSize,meshSize])
        
        #Plot decision boundaries
        CS = ax1.contour(uPlot, [likelihood_threshold], origin='lower', cmap='flag',linewidths=2, extent = [np.min(xx), np.max(xx), np.min(yy), np.max(yy)])
        CS = ax2.contour(uPlot, [likelihood_threshold], origin='lower', cmap='flag',linewidths=2, extent = [np.min(xx), np.max(xx), np.min(yy), np.max(yy)])
        plt.colorbar()
        titleString = titleVal + 'numMix = ' + str(bestMix) + 'score = ' + str(bestScore)
        plt.suptitle(titleString)
        

    return percentOutlier
    
        
        
      

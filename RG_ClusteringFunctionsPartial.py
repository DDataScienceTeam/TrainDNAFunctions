################################################################################
# Abnormality detection algorithms - 2D data usage (TrainDNA)
#
# Author: Rudraksh Goel
# Date: 14-Jul-2020
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import math

### ------------------ PYTHON FUNCTIONS TO IMPLEMENT CLUSTERING -----------------------------------

# ----------------------- Gaussian mixture models ----------------------------------------
from sklearn.mixture import GaussianMixture
from itertools import cycle, islice
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar

def gmmClustering(data, uniqueTrains):
	nComp = 1 # Number of mixtures to try 

#     threshold = 1 # Threshold by which we segment data using inter-quantile range
	bestLabels = []
	
	gmm = GaussianMixture(n_components=nComp)#, verbose=1)
	labels = gmm.fit_predict(data)
	
	score = gmm.score(data)
	scoreSamp = gmm.score_samples(data)
	
#     for i in range(4):
#         #based on quantile
# #         likelihood_threshold = np.percentile(scoreSamp, (1 - threshold)*100)
#         density = gaussian_kde(scoreSamp)
#         max_x_value = minimize_scalar(lambda x: -density(x)).x
#         print(density, max_x_value)
		
#         mean_likelihood = scoreSamp.mean()
#         new_likelihoods = scoreSamp[scoreSamp < max_x_value]
#         new_likelihoods_std = np.std(new_likelihoods)
#         likelihood_threshold = mean_likelihood - (threshold * new_likelihoods_std)
	
#         #get outliers potentially
#         prob = -gmm.score_samples(data) + likelihood_threshold
# #         print(max_x_value, prob)
		
#         fig = plt.figure(figsize = (7, 5))
#         plt.scatter(data[:,0],data[:,1],c=prob>0,s=8)
#         plt.title(f"Thresh = {threshold}")
#         plt.colorbar()
#         plt.show()
		
#         threshold += 1
	
	scoreSamp = [abs(i) for i in scoreSamp]
	av = np.median(scoreSamp)

	# tempLabels = [0 if i<5 else 1 for i in scoreSamp]
	# anomalies = [train for i, train in enumerate(uniqueTrains) if tempLabels[i]==1]
	
	tempLabels = [0 if abs(i-av)<3 else 1 for i in scoreSamp]
	anomalies = [train for i, train in enumerate(uniqueTrains) if tempLabels[i]==1]
	
	if False:
		labelColours = 'white'
		colors = np.array(['blue','red'])
		plt.scatter(data[:, 0], data[:, 1], s=20, color=colors[tempLabels], alpha=0.8)
		plt.title('Gaussian mixtures', color=labelColours)

		plt.show()
	
	return anomalies, tempLabels

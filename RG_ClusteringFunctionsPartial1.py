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

# ----------------------- K-means clustering ----------------------------------------

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as sil_score
from sklearn.metrics import calinski_harabasz_score as cal_score
from sklearn.metrics import davies_bouldin_score as dav_score

# Run kmeans clustering on the data to compare
def kmeansClustering(data, uniqueTrains):
	# Run clustering on the data
	maxClust = 8
	nClust = 2 # Start with 2 clusters and stop when the score of clusters stops improving
	
	# Metrics for the davies-bouldin score
	bestScore = 100000 # Start with an initial score 
	bestNumClust = 0
	bestKmeans = [] # Best clusters for the current time bucket
	bestCenters = []
	
	# # Metrics for the silhouette score
	# bestSilScore = 0
	# bestNClustSil = 0
	# bestKmeansSil = []
	# bestCentersSil = []

	distinct = np.array(list(set(tuple(p) for p in data))) # Get number of unique data points

	while nClust <= maxClust:
		# If there are not enough data points to form clusters, break
		if nClust >= len(data) or nClust > len(distinct):
			break

		# Try clustering on the data and evaluate the score

		# n_init refers to number of times the centroid seeds are generated for randomness/averaging of results
		kmeans = KMeans(n_clusters = nClust, n_init = 10) 
		y_kmeans = kmeans.fit_predict(data)
		
		# score = sil_score(data, kmeans.labels_, metric = 'euclidean')#, sample_size = len(data))
		davScore = dav_score(data, kmeans.labels_) 
		centers = kmeans.cluster_centers_
	
		# if False: print("Silhouette score: {:0.2f}\t Davies-Bouldin score: {:0.2f}\t Num clusters: {}".format(score, davScore, nClust))
		
		# if score > bestSilScore:
		# 	bestNClustSil = nClust
		# 	bestSilScore = score
		# 	bestKmeansSil = y_kmeans
		# 	bestCentersSil = centers 
			
		if davScore < bestScore:
			bestNumClust = nClust
			bestScore = davScore
			bestKmeans = y_kmeans
			bestCenters = centers 

		nClust += 1 # Increment the number of clusters and try to find a better score
		
	unique, counts = np.unique(bestKmeans, return_counts=True)

	
	p = 5 # Percent of total values to be considered a small cluster
	n = 5 # OR choose the number of trains to be considered a small cluster
	smallClusters = [i for i in unique if counts[i] < (p/100 * sum(counts))]
	trainID = [i for i, val in enumerate(bestKmeans) if val in smallClusters]
	anomalies = [train for ind, train in enumerate(uniqueTrains) if ind in trainID] 
	
	if False:
		labelColours = 'white'

		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
		# plt.title('Kmeans clustering')
		ax.scatter(data[:, 0], data[:, 1], c = bestKmeans, s = 50, cmap = 'viridis')
		ax.scatter(bestCenters[:, 0], bestCenters[:, 1], c = 'black', s = 200, alpha = 0.5)
		ax.set_title(f"K-means clustering. Score = {bestScore} Num clust = {bestNumClust}", color=labelColours)
		
		ax.set_xlabel('Mean normalised')#, fontsize = 15.0)
		ax.set_ylabel('Standard deviation normalised')

		ax.xaxis.label.set_color(labelColours)
		ax.yaxis.label.set_color(labelColours)
		ax.tick_params(axis='x', colors=labelColours)
		ax.tick_params(axis='y', colors=labelColours)

		# axes[1].set_title(f"Silhouette: Score = {bestSilScore} Num clust = {bestNClustSil}")
		# axes[1].scatter(data[:, 0], data[:, 1], c = bestKmeansSil, s = 50, cmap = 'viridis')
		# axes[1].scatter(bestCentersSil[:, 0], bestCentersSil[:, 1], c = 'black', s = 200, alpha = 0.5);
		# fig.tight_layout()
		
		plt.show()
	
	return anomalies, bestKmeans

# ----------------------- Local Outlier Factor ----------------------------------------
from sklearn.neighbors import LocalOutlierFactor

def lofClustering(data, uniqueTrains):
	# Run LOF on the mean and std data for the trains
	lof = LocalOutlierFactor(n_neighbors=8, contamination='auto')
	lof.fit_predict(data)
	
	# Get the negative outlier factor for the trains
	nof = lof.negative_outlier_factor_
	
	# Run thresholding on the LOF to identify anomalous trains
	labels = (nof < -2.5).astype(int)
	anomalies = [train for ind, train in enumerate(uniqueTrains) if labels[ind]==1]
	
	if False:
		f1 = plt.figure(1)
		plt.title('Negative outlier factor value for cutoff', color='white')
		plt.plot(-nof,'bo', alpha=0.3)
	#     plt.show()
		f2 = plt.figure(2)
		colours = np.array(['blue','red'])
		plt.scatter(data[:, 0], data[:, 1], s=20, color=colours[labels], alpha=0.3)
		plt.title('Local outlier factor anomalies (adjusted cutoff = 2.5)', color='white')
		plt.show()
	
	return anomalies, None

# ----------------------- Z-score method (std bracketing) ----------------------------------------
import math

def zscoreClustering(data, uniqueTrains):
	meanVals = data[:, 0]
	stdVals = data[:, 1]
	
	# Get the length of 5% of the data
	l = math.floor(5/100*np.size(meanVals))

	# Remove the top l and bottom l data points to get the standard dev/mean
	meanVals = meanVals[meanVals.argsort()][l:-l]
	stdVals = stdVals[stdVals.argsort()][l:-l]

	# Get the mean and standard deviation for the mean/std data
	meanMean = np.mean(meanVals)
	stdMean = np.std(meanVals)

	meanStd = np.mean(stdVals)
	stdStd = np.std(stdVals)

	n = 4 # Number of std away from mean required

	# Get the thresholds above and below n standard deviations away from the mean for each axis
	aboveStd = meanStd + n*stdStd
	belowStd = meanStd - n*stdStd

	aboveMean = meanMean + n*stdMean
	belowMean = meanMean - n*stdMean
	
	if False: print(f"\n std above below = {aboveStd} {belowStd}\n mean above below = {aboveMean} {belowMean}")
	
	plot = False
	if plot:
		fig = plt.figure(figsize=(8,5))
		ax = fig.add_subplot(111)
	
	anomalies = []
	for i, train in enumerate(uniqueTrains):
		currTrainMean = data[i, 0]
		currTrainStd = data[i, 1]
				
		thresh = 0
		if (currTrainMean > (aboveMean-thresh) or currTrainMean < (belowMean+thresh)) or \
					(currTrainStd > (aboveStd-thresh) or currTrainStd < (belowStd+thresh)): 
			anomalies.append(train)
			
		if plot:
			if train in anomalies:
				ax.scatter(currTrainMean, currTrainStd, s=15, label=train, c='red')
			else:
				ax.scatter(currTrainMean, currTrainStd, s=15, label=train, c='blue')

	if plot:
		labelColours = 'white'
		
		maxY, minY = max(ax.get_yticks()), min(ax.get_yticks())
		maxX, minX = max(ax.get_xticks()), min(ax.get_xticks())

		# Standard deviation axis first
		X = np.linspace(minX, maxX, 50)
		plt.fill_between(X, maxY, aboveStd, color='green', 
						 alpha=0.5) 
		plt.fill_between(X, belowStd, minY, color='green', 
						 alpha=0.5) 

		# Mean axis second
		plt.fill_between(np.linspace(minX, belowMean, 50), minY, maxY, color='green', 
						 alpha=0.5) 
		plt.fill_between(np.linspace(aboveMean, maxX, 50), minY, maxY, color='green', 
						 alpha=0.5) 

		ax.set_xlabel('Mean normalised')#, fontsize = 15.0)
		ax.set_ylabel('Standard deviation normalised')
		ax.set_title('Z-score bracketing method', color=labelColours)

		ax.xaxis.label.set_color(labelColours)
		ax.yaxis.label.set_color(labelColours)
		ax.tick_params(axis='x', colors=labelColours)
		ax.tick_params(axis='y', colors=labelColours)

		plt.show()
		
		
	return anomalies, None
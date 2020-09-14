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


# ----------------------- Information-Theoretic method (removal method) ----------------
def infoTheoClustering(data, uniqueTrains):
	# Get the mean and standard deviation for the mean/std data
	meanMean = np.mean(data[:, 0])
	meanStd = np.std(data[:, 0])

	stdMean = np.mean(data[:, 1])
	stdStd = np.std(data[:, 1])

	# CREATE CHECK FOR 0 VALUES
	
	# Create an array which notes the difference in std/mean of the set when a train is removed
	diff = np.empty((len(uniqueTrains), 4))

	for i in range(0, len(data)):
		temp = data
		temp = np.delete(temp, i, axis=0)

		# calculate difference without the value and append to diff
		diffMeanMean = np.mean(temp[:, 0]) - meanMean
		diffMeanStd = np.std(temp[:, 0]) - meanStd
		diffStdMean = np.mean(temp[:, 1]) - stdMean
		diffStdStd = np.std(temp[:, 1]) - stdStd

		diff[i] = [abs(diffMeanMean)/meanMean * 100, abs(diffMeanStd)/meanStd * 100,\
				   abs(diffStdMean)/stdMean * 100, abs(diffStdStd)/stdStd * 100]
		
	p = 5 # percentage change

	# Identify the anomalous trains:
	anomalies = uniqueTrains[[i for i, d in enumerate(diff) if any(x > p for x in d)]]
	labels = [1 if train in anomalies else 0 for train in uniqueTrains]

	if False:
		fig = plt.figure(figsize=(6, 6))
		ax = fig.add_subplot(111)

		labelColours = 'white'
		colours = np.array(['blue','red'])

		ax.scatter(data[:, 0], data[:, 1], s=20, color=colours[labels], alpha=0.8)
		ax.set_title('Information-Theoretic method anomalies', color='white')

		ax.set_xlabel('Mean normalised')#, fontsize = 15.0)
		ax.set_ylabel('Standard deviation normalised')

		ax.xaxis.label.set_color(labelColours)
		ax.yaxis.label.set_color(labelColours)
		ax.tick_params(axis='x', colors=labelColours)
		ax.tick_params(axis='y', colors=labelColours)
		plt.show()
		
	return anomalies, labels

# ----------------------- DBSCAN method ------------------------------------------------
from sklearn.cluster import DBSCAN

def dbscanClustering(data, trains, **kwargs):
	# Run DBSCAN on the mean and std data
	dbscan = DBSCAN(eps = 0.15, min_samples=1).fit(data)
	labels = dbscan.labels_
	
	# Identify small clusters
	unique, counts = np.unique(labels, return_counts = True)
	
	p = 5 # Percent of total values to be considered a small cluster
	n = 5 # OR use the number of trains which are a small cluster
	smallClusters = [i for i in unique if counts[i] < (p/100 * sum(counts))]
	trainID = [i for i, val in enumerate(labels) if val in smallClusters]

	# OPTIONAL: Use a method of bracketing here to identify if the small clusters are actually anomalous    
	anomalies = [train for ind, train in enumerate(trains) if ind in trainID] 
	tempLabels = [1 if i in anomalies else 0 for i in trains]

	anomalyDates = []
	if kwargs.get('dates', None) is not None:
		dates = kwargs.get('dates')

		anomalyDates = [date for ind, date in enumerate(dates) if ind in trainID]

	if False:
		labelColours = 'white'

		fig = plt.figure(figsize=(6, 6))
		ax = fig.add_subplot(111)

		core_samples_mask = np.zeros_like(labels, dtype=bool)
		core_samples_mask[dbscan.core_sample_indices_] = True

		# Number of clusters in labels, ignoring noise if present.
		n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
		n_noise_ = list(labels).count(-1)

		# Black removed and is used for noise instead.
		unique_labels = set(labels)
		colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

		for k, col in zip(unique_labels, colors):
			if k == -1:
				# Black used for noise.
				col = [0, 0, 0, 1]

			class_member_mask = (labels == k)

			xy = data[class_member_mask & core_samples_mask]
			plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
					 markeredgecolor='k', markersize=10)

			xy = data[class_member_mask & ~core_samples_mask]
			plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
					 markeredgecolor='k', markersize=3)

		plt.title('DBSCAN clustering output. Num clusters: %d' % n_clusters_, color=labelColours, size = 13.0)

		ax.set_xlabel('Mean normalised')#, fontsize = 15.0)
		ax.set_ylabel('Standard deviation normalised')

		ax.xaxis.label.set_color(labelColours)
		ax.yaxis.label.set_color(labelColours)
		ax.tick_params(axis='x', colors=labelColours)
		ax.tick_params(axis='y', colors=labelColours)

		plt.show()

	return anomalies, tempLabels


# ---------------------- Elliptical method (Mahalanobis method) ------------------------
from sklearn.covariance import EllipticEnvelope

np.set_printoptions(precision=3, suppress=True)

def elliptEnvMethod(data, uniqueTrains):
	

	elp = EllipticEnvelope(support_fraction = 1)
	elp.fit_predict(data)

	# Squared Mahalanobis distances of the points of data
	# Note this is the same as using the "elp.dist_" parameter
	m_d = elp.mahalanobis(data) 

	# Get the regular Mahalanobis distances
	elp_d = np.sqrt(m_d)

	# IMPLEMENT THE AUTOMATED CUT-OFF 
	SCORE_INCREASE_RATIO = 1.3

	sortD = np.sort(elp_d)
	sortD = sortD[math.floor(len(sortD)/2):] # Get the end half of the sorted list of scores

	ratioD = np.array([sortD[i]/sortD[i-1] for i in range(1, len(sortD))])

	# print(f'\nSorted distances: {sortD}\n\n Ratios: {ratioD}')

	ind = np.where(ratioD > SCORE_INCREASE_RATIO)

	if len(ind[0]) >= 1:
		ind = ind[0][0] + 1
		SIGMA = sortD[ind] # Get the score which increases by the score_ratio compared to the previous score
	else:
		SIGMA = 100.0 # use an arbritary high score as there are no big score jumps

	# print(f'old=4, new={SIGMA}')

	SIGMA = max(SIGMA, 4.0) # Limit the SIGMA function from being too low

	# Segment
	labels = (elp_d >= SIGMA).astype(int)
	labelOLD = (elp_d > 4.0).astype(int)

	if False: 
		print(f'.dist_ = {m1}\t .m(data)={m2}')#\n {m_d}')
		print(f"Sigma labels: {labels}")
		print(f"\nCovariance = {elp.covariance_}")

	if False:
		labelColours = 'white'

		fig, ax = plt.subplots(nrows=1, ncols=3, figsize = (16, 6))
		colours = np.array(['blue','red'])

		ax[0].set_title('Elliptical envelope - Mahalanobis distance', color='white')
		ax[0].plot(elp_d,'bo', alpha=0.4)#, color=colours[labels])
		
		ax[1].scatter(data[:, 0], data[:, 1], s=20, color=colours[labels], alpha=0.4)
		ax[1].set_title('Elliptical envelope (adjusted cutoff={})'.format(SIGMA), color='white')

		ax[2].scatter(data[:, 0], data[:, 1], s=20, color=colours[labelOLD], alpha=0.4)
		ax[2].set_title('Elliptical envelope (adjusted cutoff={})'.format(4.0), color='white')

		for i, a in enumerate(ax.flat):
			ax[i].set_xlabel('Mean normalised')#, fontsize = 15.0)
			ax[i].set_ylabel('Standard deviation normalised')

			ax[i].xaxis.label.set_color(labelColours)
			ax[i].yaxis.label.set_color(labelColours)
			ax[i].tick_params(axis='x', colors=labelColours)
			ax[i].tick_params(axis='y', colors=labelColours)

		# fig1 = plt.figure(figsize=(5,5))


		plt.show()
		
	anomalies = [train for ind, train in enumerate(uniqueTrains) if labels[ind]==1]

	return anomalies, labels
	
# ---------------------- Rudy method (Euclidean distance from centroid) ------------------------
# from scipy.stats import kurtosis, jarque_bera
# from scipy.spatial.distance import euclidean as euc

# def rudyMethod(data, uniqueTrains):
	
# 	# find the centroid of the data
# 	length = data.shape[0]
# 	centroid = [sum(data[:, 0])/length, sum(data[:, 1])/length]

# 	# get the kurtosis score
# 	kurt = kurtosis(data)
# 	j = [jarque_bera(data[:, 0])[0], jarque_bera(data[:, 1])[0]]

# 	# get the outliers
	
	
# 	# plot the figure with outliers
# 	f1 = plt.figure(figsize=(5, 5))
# 	ax = plt.subplot(111)

# 	colours =  np.array(['blue','red'])
# 	ax.scatter(data[:, 0], data[:, 1], color='blue', s=8)
# 	ax.scatter(centroid[0], centroid[1], color='green', s=40)

# 	# circ = plt.Circle(tuple(centroid), 2/(kurt[0]/32), color='black', fill=False)
# 	# ax.add_artist(circ)

# 	print(f"Centroid = {centroid} kurtosis = {kurt} JB = {j}")#" thresh = {thresh}")#" outlier = {outlier}")
# 	plt.show()
	
# 	return None, None
	

# ---------------------- OPTICS method (extension of dbscan technique) ------------------------
from sklearn.cluster import OPTICS

def opticMethod(data, uniqueTrains):
	opt = OPTICS(min_samples=2, cluster_method='dbscan', eps = 0.1)#, max_eps=0.5)
	labels = opt.fit_predict(data)

	print(labels)

	if False:
		colors = ['C'+str(i+1) for i in labels]

		f = plt.figure()
		plt.scatter(data[:, 0], data[:, 1], s=20, color=colors, alpha=0.5)
		plt.show()


	return None, None
	

# ---------------------- Non-parametric method (use of histogram thresholding) ------------------------
from scipy.spatial.distance import euclidean as euc

def histogramMethod(data, uniqueTrains):

	####### Method #1: use the euclidean distance to the centroid for each point and discretise that ######

	# find the centroid of the data		
	length = data.shape[0]

	# use the middle 90% of the data to get the value of the centroid 

	# Get the length of 5% of the data
	l = math.floor(5/100*np.size(data[:, 0]))

	# Remove the top l and bottom l data points to get the standard dev/mean
	m90 = data[:, 0][data[:, 0].argsort()][l:-l]
	s90 = data[:, 1][data[:, 1].argsort()][l:-l]

	centroid = [sum(data[:, 0])/length, sum(data[:, 1])/length]
	centroid90 = [sum(m90)/length, sum(s90)/length]

	# get euclidean distance to centroid for each point
	dists = {}
	for i, val in enumerate(data):
		dists[i] = euc(centroid90, val)

	# create a histogram distribution for the data
	h, e = np.histogram(list(dists.values()), bins=10)
	# print(h, e)

	p = 5 # Percent of total values to be considered a small cluster

	# Get the anomalies based on distance away from centroid
	anomBins = [e[i] for i, val in enumerate(h) if val < (p/100 * len(uniqueTrains)) and val > 0]
	# print(anomBins)

	# Get the anomalous trains
	anomInds = [key for key, val in dists.items() for i in anomBins if (val >= i)]
	anomInds = list(set(anomInds))
	anomalies = uniqueTrains[anomInds]

	# print(sorted(anomalies))

	labels = [1 if train in anomalies else 0 for train in uniqueTrains]

	if False:
		colors = np.array(['blue','red'])

		f2 = plt.figure()
		plt.scatter(data[:, 0], data[:, 1], s=20, color=colors[labels], alpha=0.5)
		plt.scatter(centroid90[0], centroid90[1], color='green', s=40)
		plt.show()

	return anomalies, labels

# ---------------------- Non-parametric method (use of 2d histogram thresholding) ------------------------

def histogram2DMethod(data, uniqueTrains):

	####### Method 2: Use both axes and a simple digitising method - 2D histogram ######

	# # Get the x and y axes data
	xVals, yVals = data[:, 0], data[:, 1]

	nBins = 4

	# Histogram in 2 dimensions, using a small number of bins 
	H, edges = np.histogramdd(data, bins = nBins)

	# print(f'Hist values = {H}\t Edges={edges}')

	# Get the values which are less than p percent
	p = 5 # Percent of total values to be considered a small cluster

	# Get indexes of bins where less than p% of points
	inds = np.where((H < (p/100 * len(uniqueTrains))) & (H > 0)) 
	inds = list(zip(inds[0], inds[1]))
	# print(inds)

	# Find row indexes where value is within the range
	rowInds = []
	for val in inds:
		xmin, xmax = edges[0][val[0]], edges[0][val[0]+1]	
		ymin, ymax = edges[1][val[1]], edges[1][val[1]+1]

		t1 = data[:, 0] >= xmin
		t2 = data[:, 0] <= xmax
		t3 = data[:, 1] >= ymin
		t4 = data[:, 1] <= ymax

		dataPoints = np.where(t1 & t2 & t3 & t4)
		rowInds.extend(dataPoints[0].tolist())

	anomalies = uniqueTrains[rowInds]
	# print(np.sort(anomalies))

	labels = [1 if train in anomalies else 0 for train in uniqueTrains]

	# Plot the histogram in 2D if required
	if False:

		labelColours = 'white'
		colors = np.array(['blue','red'])

		f2 = plt.figure()
		ax1 = f2.add_subplot(111)
		ax1.scatter(data[:, 0], data[:, 1], s=20, color=colors[labels])#, alpha=0.8)
		ax1.set_xlabel('Mean normalised')#, fontsize = 15.0)
		ax1.set_ylabel('Standard deviation normalised')

		ax1.set_title('Histogram anomalies', color=labelColours)
		ax1.xaxis.label.set_color(labelColours)
		ax1.yaxis.label.set_color(labelColours)
		ax1.tick_params(axis='x', colors=labelColours)
		ax1.tick_params(axis='y', colors=labelColours)

		f3 = plt.figure()
		plt.hist2d(xVals, yVals, bins = nBins)#, bins='auto')		
		plt.colorbar()
		plt.show()

	return anomalies, labels

# ---------------------- Isolation forest method ---------------------------------------------
from sklearn.ensemble import IsolationForest

def isolationForest(data, uniqueTrains):

	clf = IsolationForest(n_estimators=100, behaviour="new", warm_start=True, contamination=0.03)#, max_samples='auto')
	tempLabels = clf.fit_predict(data)

	if False:
		score_samples = clf.score_samples(data)
		df = clf.decision_function(data)

		print(f'Anomaly scores = {df}\n')
		print(f'Labels = {labels}')
		print(f'\nScore samples = {score_samples}')


	labels = [1 if val==-1 else 0 for val in tempLabels]

	anomalies = [train for ind, train in enumerate(uniqueTrains) if labels[ind]==1]

	print(sorted(anomalies))

	# Plot the isolation forest anomaly output
	if False:
		colors = np.array(['blue','red'])

		f2 = plt.figure()
		plt.title("Isolation forest anomalies")
		plt.scatter(data[:, 0], data[:, 1], s=20, color=colors[labels], alpha=0.5)
		plt.show()

	return anomalies, labels



# --------------------------------------------------------------------------------------------------------
# ---------------------- METHODS TO CHECK NORMALNESS OF DATA ---------------------------------------------
# --------------------------------------------------------------------------------------------------------

from scipy.stats import kurtosis, jarque_bera

# Defines the normalness of the data on both axes
def normalnessTest(data, plot=True):
	#******** Get the kurtosis score using all of the data ********
	k = kurtosis(data) 
	
	#******** Get the kurtosis score for 90% of the data **********
	m = data[:, 0]
	s = data[:, 1]
	
	# Get the length of 5% of the data
	l = math.floor(5/100*np.size(m))

	# Remove the top l and bottom l data points to get the standard dev/mean
	m90 = m[m.argsort()][l:-l]
	s90 = s[s.argsort()][l:-l]
	
	k1_90 = kurtosis(m90)
	k2_90 = kurtosis(s90)
	
	k90 = [k1_90, k2_90]
	
	#******************* Get the Jarque-Bera score *****************
	j1 = jarque_bera(m)
	j2 = jarque_bera(s)
	
	j = [j1[0], j2[0]]
	
	#********* Get the Jarque-Bera score for 90% of the data ********
	j1_90 = jarque_bera(m90)[0]
	j2_90 = jarque_bera(s90)[0]
	
	j90 = [j1_90, j2_90]
	
	print(f'Kurtosis\t {k}\nKurtosis (90%)\t {k90}\nJarque-Bera\t {j}\nJB (90%)\t {j90}')




# --------------------------------------------------------------------------------------------------------
# #---- Example of anomaly detection on generated data ------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
import matplotlib.colors as mcolors

def implementSampleData():
	# 72 trains with one data point each 10 minutes for 1 day of data
	x = 100 * np.random.random(size=(72, 24*6)) # Distribute it between 0-100

	# Add an outlier which is only 0-80
	x[0:1,:] = 80 * np.random.random(size=(1, 24*6))

	train_m = np.mean(x,axis=1)
	train_sd = np.std(x,axis=1)

	data = np.array((train_m, train_sd))
	data = np.transpose(data)
	data = MinMaxScaler().fit_transform(data)

	algorithms = {'DBSCAN Clustering': dbscanClustering,'Info-Theo Method':infoTheoClustering,\
				  'Z-Score Bracketing': zscoreClustering, 'LOF Clustering': lofClustering, \
				  "Gaussian Mixtures": gmmClustering, 'kMeans Clustering': kmeansClustering, \
				 "Mahalanobis Method": elliptEnvMethod}#, 'Histogram Method': histogramMethod}

	trainSet = np.array(['A'+str(i) for i in range(0, 72)]) # Generate a random set of trains

	allAnoms = []

	for alg in algorithms:
		# try:
		anomalies = []
		anomalies, labels = algorithms[alg](data, trainSet)
		# except Exception as e:
			# print(f'Failed for {alg}\n')
			# print(e)
			
		allAnoms.extend(anomalies)
		
		print("Algorithm: {} \t Anomalies: {}".format(alg, sorted(anomalies)))

	anomUnique, anomCounts = np.unique(allAnoms, return_counts = True)
	trueAnoms = [anomUnique[i] for i, val in enumerate(anomCounts) if (val >= 0.5*len(algorithms))]
	print(f"\nTrue anomalies with voting: {trueAnoms}\n")

	# plot if needed 
	if True:
		labelColours = 'white'
		f, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

		for i in np.arange(0, 71, 1):
			ax[0].plot(x[1:,:][i], c='blue', alpha=0.2)   
		ax[0].plot(x[0:1,:][0], c='red')

		ax[0].set_title('Time series of random sample data', color=labelColours)
		ax[0].set_xlabel('Time bucket (1 = 10 minutes)')
		ax[0].set_ylabel('Simulated signal value (normalised)')

		# colours = np.array(['blue','red'])
		ax[1].scatter(data[:, 0], data[:, 1], s=20, alpha=0.4)

		ax[1].set_title('Sample data in range 0-100 and an outlier in range 0-80', color=labelColours)
		ax[1].set_xlabel('Series mean')
		ax[1].set_ylabel('Series standard deviation')

		# ax[1].set_title('Histogram anomalies', )
		for i, a in enumerate(ax.flat):
			ax[i].xaxis.label.set_color(labelColours)
			ax[i].yaxis.label.set_color(labelColours)
			ax[i].tick_params(axis='x', colors=labelColours)
			ax[i].tick_params(axis='y', colors=labelColours)

		plt.show()

		col = mcolors.TABLEAU_COLORS

		try:
			del col['tab:blue']
		except:
			pass
		col = list(col.items())
		colCount = 0
		

		fig, axe = plt.subplots(1, 1, figsize=(6, 6))
			
		for i, train in enumerate(trainSet):
			
			if train in trueAnoms:
				axe.scatter(data[i, 0], data[i, 1], c=col[colCount][1],\
								label=train, s=20, alpha=0.4)
				colCount +=1
			else:
				axe.scatter(data[i, 0], data[i, 1], label='_nolegend_', \
								c='blue', s=20, alpha=0.4)

		axe.set_xlabel('Mean normalised')#, fontsize = 15.0)
		axe.set_ylabel('Standard deviation normalised')
		axe.set_title(f'Output of anomaly detection using {len(algorithms)} algorithms', color=labelColours)

		axe.xaxis.label.set_color(labelColours)
		axe.yaxis.label.set_color(labelColours)
		axe.tick_params(axis='x', colors=labelColours)
		axe.tick_params(axis='y', colors=labelColours)
		axe.legend(loc='upper left', fancybox=True, framealpha=1, bbox_to_anchor=(1,1.02))
		
		plt.show()
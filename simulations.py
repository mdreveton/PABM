#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:03:41 2024

@author: maximilien
"""

import networkx as nx
import numpy as np
import scipy as sp

import utils as utils
import clustering as clustering
import crossvalidation as cv


"""
n = 900
n_clusters = 3

sizes = [ n//n_clusters for dummy in range( n_clusters ) ]

labels_true = [ ]
for community in range( n_clusters ):
    labels_true = labels_true + [ community+1 for i in range( sizes[community] ) ]
labels_true = np.array( labels_true, dtype = int )

theta_in = np.random.lognormal( mean = 1, sigma = 1, size = n )
theta_out = np.random.lognormal( mean = 1, sigma = 1, size = n )
theta_out = np.ones( n )

theta_in = np.random.beta( 2, 2, size = n )
theta_out = theta_in
p = 0.05
q = 0.01
P = utils.generateP_of_homogeneousPABM( labels_true, p, q, theta_in, theta_out )

A = utils.generateBernoulliAdjacency( P )

z_bm = clustering.spectralClustering_bm( A , n_clusters )
z_dcbm = clustering.spectralClustering_dcbm( A , n_clusters )
z_pabm = clustering.spectralClustering_pabm( A, n_clusters )


print( 'Accuracy with SBM algo : ' , utils.computeAccuracy( labels_true, z_bm ) )
print( 'Accuracy with DC-SBM algo : ' , utils.computeAccuracy( labels_true, z_dcbm ) )
print( 'Accuracy with PA-SBM algo : ' , utils.computeAccuracy( labels_true, z_pabm ) )

cv.crossValidation_knownNumberOfClusters( A, n_clusters=n_clusters, epsilon=0.15)



z_bm, P_bm = clustering.clustering_bm( A, n_clusters )
z_dcbm, P_dcbm = clustering.clustering_dcbm( A, n_clusters )
z_pabm, P_pabm = clustering.clustering_pabm( A, n_clusters )


"""


"""

Lambdas = [ [] for a in range(n_clusters) ]
for a in range( n_clusters ):
    #Lambdas[ a ] = [ np.random.lognormal( mean = 1, sigma = 1, size = sizes[ a ] ) for b in range(n_clusters) ]
    Lambdas[ a ] = [ np.random.beta( 2, 2, size = sizes[ a ] ) for b in range(n_clusters) ]


P_clusterRowNoralized = P.copy()
nodeListPerCommunity = utils.obtain_community_lists( labels_true, n_clusters = len( sizes ) )

for i in range( n ):
    normalisation = np.zeros( n_clusters )
    for a in range( n_clusters ):
        normalisation[ a ] = np.linalg.norm( P[ [i], nodeListPerCommunity[ a ] ], ord = 1)
    for j in range( n ):
        P_clusterRowNoralized[ i, j ] = P[ i, j ] / normalisation[ labels_true[j] - 1 ]
    
"""


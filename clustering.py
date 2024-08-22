#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:31:02 2024

@author: dreveton
"""

import numpy as np
import scipy as sp
from sklearn.cluster import KMeans, SpectralClustering

import utils as utils

from tqdm import tqdm



def clustering_bm( A, n_clusters, n_iter = 10 ):
    
    z = spectralClustering_bm( A, n_clusters )
    
    for iteration in range(n_iter ):
        z, B = likelihoodImprovement_bm( A, n_clusters, z )
    
    if n_iter == 0:
        node_in_each_clusters = utils.obtain_community_lists( z, n_clusters = n_clusters )
        B = estimateRateMatrix_bm( A, n_clusters, node_in_each_clusters )

    Z = utils.oneHotEncoding( z, n_clusters = n_clusters )
    P_hat = Z @ B @ Z.T
    
    return z.astype(int), P_hat


def clustering_dcbm( A, n_clusters, n_iter = 10 ):
    
    z = spectralClustering_dcbm( A, n_clusters )
    for iteration in tqdm( range( n_iter ) ):
        z, P_hat = likelihoodImprovement_dcbm( A, n_clusters, z )
    
    if n_iter == 0:
        node_in_each_clusters = utils.obtain_community_lists( z, n_clusters = n_clusters )
        edge_count = edge_count_between_communities( A, n_clusters, node_in_each_clusters )
        theta_hat = estimate_theta_dcbm( A, z, edge_count )                
        Z = utils.oneHotEncoding( z, n_clusters )
        P_hat = np.diag( theta_hat ) @ Z @ edge_count @ Z.T @ np.diag( theta_hat )
    
    return z.astype(int), P_hat



def clustering_pabm( A, n_clusters, n_iter = 10 ):
    
    z = spectralClustering_pabm( A, n_clusters )
    for iteration in tqdm( range( n_iter ) ):
        z, lambda_hat = likelihoodImprovement_pabm( A, n_clusters, z )
        
    if n_iter == 0:
        lambda_hat = 0
    
    n = A.shape[0]
    P_hat = np.zeros( (n,n) )
    for i in range( n ):
        for j in range( n ):
            P_hat[i,j] = lambda_hat[ i, z[j]-1 ] * lambda_hat[ j, z[i]-1 ]

    return z.astype(int), P_hat



# =============================================================================
# SPECTRAL CLUSTERING: VARIOUS FORMS
# =============================================================================

def spectralClustering_bm( A , n_clusters ):
    """ Perform spectral clustering for a SBM
    Parameters
    ----------
    A : {array-like, sparse matrix} of shape (n, n)
        Adjacency matrix of a graph.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    z : vector of shape (n) with integer entries between 1 and n_clusters
        entries z_i denotes the cluster of vertex i.
    """
    vals, vecs = sp.sparse.linalg.eigsh( A.astype(float), k = n_clusters, which = 'LM' )

    hatP = vecs @ np.diag( vals ) @ vecs.T
    z = KMeans(n_clusters = n_clusters, n_init = 'auto' ).fit_predict( hatP ) + np.ones( A.shape[0] )
    
    return z.astype(int) 


def spectralClustering_dcbm( A , n_clusters ):
    """ Perform spectral clustering for a DCBM
    Parameters
    ----------
    A : {array-like, sparse matrix} of shape (n, n)
        Adjacency matrix of a graph.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    z : vector of shape (n) with integer entries between 1 and n_clusters
        entries z_i denotes the cluster of vertex i.
    """
    n = A.shape[0]
    vals, vecs = sp.sparse.linalg.eigsh( A.astype(float), k = n_clusters, which = 'LM' )
    #vals, vecs = sp.sparse.linalg.eigsh( A.astype(float), k = n_clusters * n_clusters, which = 'BE' )

    
    hatP = vecs @ np.diag( vals ) @ vecs.T
    hatP_rowNormalized = hatP
    for i in range( n ):
        hatP_rowNormalized[i,:] = hatP[i,:] / np.linalg.norm( hatP[i,:], ord = 1)
    
    z = KMeans(n_clusters = n_clusters, n_init = 'auto' ).fit_predict( hatP_rowNormalized ) + np.ones( n )
    
    return z.astype(int) 


def spectralClustering_pabm( A, n_clusters ):
    """ Perform spectral clustering for a PABM
    Parameters
    ----------
    A : {array-like, sparse matrix} of shape (n, n)
        Adjacency matrix of a graph.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    z : vector of shape (n) with integer entries between 1 and n_clusters
        entries z_i denotes the cluster of vertex i.
    """
    
    n = A.shape[0]
    vals, vecs = sp.sparse.linalg.eigsh( A.astype(float), k = n_clusters * n_clusters, which = 'BE' )
    
    hatP = vecs @ np.diag( vals ) @ vecs.T
    hatP_rowNormalized = hatP
    for i in range( n ):
        hatP_rowNormalized[i,:] = hatP[i,:] / np.linalg.norm( hatP[i,:], ord = 1)
    
    z = KMeans(n_clusters = n_clusters, n_init = 'auto' ).fit_predict( hatP_rowNormalized ) + np.ones( n )
    
    return z.astype(int) 


def orthogonalSpectralClustering( A, n_clusters ):
    """Perform spectral clustering (Algorithm 2 of Koo el al.)
    Parameters
    ----------
    A : {array-like, sparse matrix} of shape (n, n)
        Adjacency matrix of a graph.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    z : vector of shape (n) with integer entries between 1 and n_clusters
        entries z_i denotes the cluster of vertex i.
    """
    vals, vecs = sp.sparse.linalg.eigsh( A.astype(float), k = n_clusters * n_clusters, which = 'BE' )
    #hatP = vecs @ np.diag( vals ) @ vecs.T

    n = A.shape[0]
    B = n * vecs @ vecs.T
    
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed').fit( np.abs(B) )
    z = clustering.labels_ + np.ones( n )

    return z.astype(int) 




# =============================================================================
# BM: LIKELIHOOD IMPROVEMENT AND PARAMETER ESTIMATIONS
# =============================================================================

def estimateRateMatrix_bm( A, n_clusters, node_in_each_clusters ):
    
    B = np.zeros( (n_clusters, n_clusters) )
    
    for a in range( n_clusters ):
        for b in range( a + 1 ):
            dummy = A[ node_in_each_clusters[ a ], : ]
            dummy = dummy[ :, node_in_each_clusters[ b ] ]
            if a != b:                
                normalisation = len( node_in_each_clusters[ a ] ) * len( node_in_each_clusters[ b ] )
            else:
                normalisation = len( node_in_each_clusters[ a ] ) * ( len( node_in_each_clusters[ a ] ) - 1 )
            
            B[ a, b ] = np.sum( dummy ) / normalisation
            B[ b, a ] = B[ a, b ]
    
    return B


def number_neighbors_in_each_community( A, n_clusters, z, node_in_each_clusters ):
    n = A.shape[0]
    number_neighbors = np.zeros( ( n, n_clusters ) )
    for a in range(n_clusters):
        dummy = A[:,node_in_each_clusters[a] ]
        for i in range( n ):
            number_neighbors[i,a] = np.sum( dummy[ [i], : ] )

    return number_neighbors


def likelihoodImprovement_bm( A, n_clusters, z_init ):
    
    n = A.shape[0]
    
    z = np.zeros( n )
    
    node_in_each_clusters = utils.obtain_community_lists( z_init, n_clusters = n_clusters )
    B = estimateRateMatrix_bm( A, n_clusters, node_in_each_clusters )
    
    number_neighbors = number_neighbors_in_each_community( A, n_clusters, z, node_in_each_clusters ) 
    
    for i in range( n ):
        Li = np.zeros( n_clusters )
        for a in range( n_clusters ):
            Li[ a ] = np.sum( [ number_neighbors[i,b] * np.log( B[a,b] ) for b in range( n_clusters ) ] ) 
            Li[a] += np.sum( [ ( len( node_in_each_clusters[b] ) - number_neighbors[i,b] ) * np.log( 1-B[a,b] ) for b in range( n_clusters ) ] )
        z[ i ] = np.argmax( Li ) + 1
    
    return z.astype(int), B



# =============================================================================
# DC-BM: LIKELIHOOD IMPROVEMENT AND PARAMETER ESTIMATIONS
# =============================================================================

def edge_count_between_communities( A, n_clusters, node_in_each_clusters ):
    edge_count = np.zeros( ( n_clusters, n_clusters ) )
    for a in range( n_clusters ):
        dummy = A[:,node_in_each_clusters[a] ]
        for b in range( a+1 ):
            edge_count[a,b] = dummy[node_in_each_clusters[b],:].sum()
            edge_count[b,a] = edge_count[a,b]
            
    return edge_count


def estimate_theta_dcbm( A, z, edge_count ):
    n = A.shape[ 0 ]
    theta_hat = np.zeros( n )
    for i in range( n ):
        theta_hat[ i ] = A[[i], : ].sum() / np.sum( edge_count[ z[i] - 1, : ] )

    return theta_hat


def likelihoodImprovement_dcbm( A, n_clusters, z_init, tol = 0.0000000001 ):
    
    n = A.shape[0]
    z = np.zeros( n )
    
    node_in_each_clusters = utils.obtain_community_lists( z_init, n_clusters = n_clusters )
    edge_count = edge_count_between_communities( A, n_clusters, node_in_each_clusters )
    theta_hat = estimate_theta_dcbm( A, z_init, edge_count )            
    
    Z = utils.oneHotEncoding( z_init, n_clusters )
    P_hat = np.diag( theta_hat ) @ Z @ edge_count @ Z.T @ np.diag( theta_hat )
    
    for i in range( n ):
        Li = np.zeros( n_clusters )
        for a in range( n_clusters ):
            Li[a] = np.sum( [ A[i,j] * np.log( min( 1, theta_hat[i] * theta_hat[j] * edge_count[ a, z_init[j] - 1 ] + tol ) ) + (1-A[i,j]) * np.log( max( tol, 1 - theta_hat[i] * theta_hat[j] * edge_count[ a, z_init[j]-1 ] ) ) for j in range( n ) ] )
        z[ i ] = np.argmax( Li ) + 1
    
    return z.astype(int), P_hat



# =============================================================================
# PA-BM: LIKELIHOOD IMPROVEMENT AND PARAMETER ESTIMATIONS
# =============================================================================

def likelihoodImprovement_pabm( A, n_clusters, z_init, tol = 0.0000000001 ):

    n = A.shape[0]
    z = np.zeros( n )
    
    node_in_each_clusters = utils.obtain_community_lists( z_init, n_clusters = n_clusters )
    number_neighbors = number_neighbors_in_each_community( A, n_clusters, z, node_in_each_clusters ) 
    edge_count = edge_count_between_communities( A, n_clusters, node_in_each_clusters )

    lambda_hat = np.zeros( ( n, n_clusters ) )
    for i in range( n ):
        for a in range( n_clusters ):
            lambda_hat[ i, a ] = number_neighbors[ i, a ] / np.sqrt( edge_count[ a, z_init[ i ] - 1 ] )
    
    for i in range( n ):
        Li = np.zeros( n_clusters )
        
        for a in range( n_clusters ):
            Li[a] = np.sum ( [ number_neighbors[ i,b ] * np.log ( number_neighbors[ i,b ] / np.sqrt( edge_count[i,b] ) ) for b in range( n_clusters ) ] )
            #Li[ a ] = np.sum( [ A[i,j] * np.log( min(1, lambda_hat[i,z_init[j]-1] * lambda_hat[j,a] + tol ) ) + (1-A[i,j]) * np.log( max(tol, 1 - lambda_hat[i,z_init[j]-1] * lambda_hat[j,a] ) ) for j in range( n ) ] )
        
        z[ i ] = np.argmax( Li ) + 1
    
    return z.astype(int), lambda_hat


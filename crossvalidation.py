#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 18:27:17 2024

@author: maximilien
"""

import numpy as np
#import networkx as nx
import igraph as ig

import clustering as clustering 


def mask( n, epsilon, directed = False ):
    """
    Parameters
    ----------
    n : TYPE
        DESCRIPTION.
    epsilon : TYPE
        DESCRIPTION.
    directed : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    array in scipy sparse csr format
        DESCRIPTION.

    """
    #G = nx.fast_gnp_random_graph(n, epsilon, directed = directed )
    G = ig.Graph.Erdos_Renyi( n , epsilon, directed = directed, loops = False )
    
    return G.get_adjacency_sparse( )


def l2loss( A, P, mask_test ):
    """
    
    """
    
    if A.shape != P.shape:
        raise TypeError( 'The matrices A and P should be of same shape' )
    if A.shape != mask_test.shape or P.shape != mask_test.shape :
        raise TypeError( 'The mask matrix should be of same shape as A and P' )
        
    return np.linalg.norm( np.multiply (A-P, mask_test ) , ord='fro' ) 
    

def crossValidation_knownNumberOfClusters( A, n_clusters, epsilon, modelType = 'sbm', directed = False ):
    
    if modelType.lower() not in [ 'sbm', 'dcbm', 'pabm' ]:
        raise TypeError( 'The model is not implemented' )
    
    if A.shape[0] != A.shape[1]:
        raise TypeError( 'A should be a square matrix' )
    if np.min( A ) < 0:
        raise TypeError( 'A should only have non-negative entries' )
    
    n = A.shape[0]
    
    Mtest = mask( n, epsilon, directed=directed )
    Mtest = Mtest.todense( )
    Mtrain = np.ones( (n,n) ) - Mtest 
    
    Atrain = A * Mtrain #entrywise matrix multiplication
    
    P_bm, z_sbm = clustering.spectralClustering_BM( A, n_clusters = n_clusters )
    P_dcbm, z_dcbm = clustering.spectralClustering_DCBM( A, n_clusters = n_clusters )
    P_pabm, z_pabm = clustering.spectralClustering_PABM( A, n_clusters = n_clusters )
    
    Atest = A * Mtest #entrywise matrix multiplication
    
    elll2loss = { 'bm' : 0,
             'dcbm' : 0,
             'pabm' : 0 }
    
    elll2loss[ 'bm' ] = l2loss( Atest, P_bm, Mtest )
    elll2loss[ 'dcbm' ] = l2loss( Atest, P_dcbm, Mtest )
    elll2loss[ 'pabm' ] = l2loss( Atest, P_pabm, Mtest )

    
    return elll2loss
    
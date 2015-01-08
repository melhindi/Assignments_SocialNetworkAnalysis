"""
Performs evolutionary clustering based on graph snapshots given as edge lists
Requires python-igraph, numpy, scipy, matplotlib
"""
__author__ = 'melhindi'

import sys
import os
from pprint import pprint
import logging
from igraph import *
import numpy as np
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt


def perform_spectral_clustering_onGraph(graph, k, normalized=True, num_vectors_kMeans=3):
    """
    Performs spectral clustering on an given igraph Graph object

    :param graph: igraph Graph object
    :param k: Number of communities/clusters to detect
    :param normalized: Shall the normalized laplacian be used for clustering? Defaults to True
    :param num_vectors_kMeans: The number of top eigenvectors to use for clustering. Defaults to 3
    :return: (membership_matrix, sse) Tupel consisting of a membership matrix as numpy array and
                                        the sum of squared errors of the clustering
    """

    # Use igraph method to compute laplacian
    laplacian = np.array(graph.laplacian(normalized=normalized))
    return perform_spectral_clustering_onLaplacian(laplacian, k)


def perform_spectral_clustering_onLaplacian(laplacian, k, num_vectors_kMeans=3, considerZeroEigenvalues=False):
    """
    Performs spectral clustering based on a given graph laplacian passed as numpy array

    :param laplacian: Numpy array representing a graph laplacian
    :param k: Number of communities/clusters to detect
    :param num_vectors_kMeans: The number of top eigenvectors to use for clustering. Defaults to 3
    :return: (membership_matrix, sse) Tupel consisting of a membership matrix as numpy array and
                                        the sum of squared errors of the clustering
    """
    np.set_printoptions(precision=3, linewidth=200)

    num_vertices = laplacian.shape[0]

    eigenValues, eigenVectors = np.linalg.eig(laplacian)

    #Sort eigenvalues ascending
    idx = eigenValues.argsort() # this returns the indexes of the values
    eigenValues = eigenValues[idx] # sort eigenvalues
    eigenVectors = eigenVectors[:, idx] #sort eigenvectors

    clustering_eigenValues_index = []
    # Prepare to pick num_vectors_kMeans smallest eigenvalues
    if (considerZeroEigenvalues):
        clustering_eigenValues_index = range(num_vectors_kMeans) #0,...,num_vectors_kMeans-1
    else:
        index = 0
        while len(clustering_eigenValues_index) < num_vectors_kMeans and index < len(eigenValues):
            if eigenValues[index] >= 1e-10:
                clustering_eigenValues_index.append(index)
            index += 1

    logging.debug("Normalized Graph Laplacian")
    logging.debug(laplacian)
    logging.debug("EigenValues")
    logging.debug(eigenValues)
    logging.debug("EigenVectors")
    logging.debug(eigenVectors)

    print "debug"
    print clustering_eigenValues_index
    data = eigenVectors[:, clustering_eigenValues_index]

    logging.info("Top %i eigenvectors used for clustering" % num_vectors_kMeans)
    logging.info(data)


    # Do k-Means clustering on top num_vectors_kMeans eigenvectors
    centroids, labels = kmeans2(data, k)

    logging.debug("Centroids")
    logging.debug(centroids)
    logging.debug("Labels")
    logging.debug(labels)

    sse = compute_SSE(data, centroids, labels)

    # Generate membership matrix
    membership_matrix = np.zeros((num_vertices, k))
    for i in xrange(num_vertices):
        membership_matrix[i][labels[i]] = 1

    logging.debug("Membership matrix")
    logging.debug(membership_matrix)

    return (membership_matrix, sse)


def compute_SSE(data, centroids, labels):
    """
    Compute the sum of squared errors for a given clustering

    :param data: Data points as (rxn) numpy array, where each row r represents a data point with n dimensions
    :param centroids: Centroids as (kxn) numpy array, where each row k represents a centroid with n dimensions
    :param labels: Numpy array label[i] is the code or index of the centroid the i'th observation is closest to.
    :return: value of sse as number
    """
    sse = 0
    for point in xrange(len(labels)):
        vector = data[point, :]
        centroid = centroids[labels[point]]
        sse += np.sum((vector - centroid) ** 2)

    return sse


def perform_evolutionary_clustering(graph, prev_membership_matrix, alpha, k):
    """
    Performs evolutionary clustering for a given graph at time t and the membership matrix of time t-1

    :param graph: igraph Graph object
    :param prev_membership_matrix: numpy array representing the membership matrix with vertices as rows and
                                                                            commuinities/clusters as columns
    :param alpha: Alpha parameter of evolutionary clustering
    :param k: Number of communities/clusters to detect
    :return: (membership_matrix, sse) Tupel consisting of a membership matrix as numpy array and
                                        the sum of squared errors of the clustering
    """
    laplacian_hat = np.array(graph.laplacian(normalized=True))
    laplacian_hat[laplacian_hat != 1] *= alpha
    mm_transposed = np.transpose(prev_membership_matrix)
    L_hat = laplacian_hat - (1 - alpha) * prev_membership_matrix.dot(mm_transposed)

    return perform_spectral_clustering_onLaplacian(L_hat, k,considerZeroEigenvalues=True)


if __name__ == "__main__":
    # File is called as script
    # input to script: multiple edge lists representing the different snapshots
    # <edgelist1> <edgelist2> ... <edgelistN> files are assumed to be given in correct order


    #Parse mandatory command line parameters
    if len(sys.argv) < 2:
        script_name = os.path.basename(__file__)
        print "Usage: " + str(script_name) + " <log_level> <edgelist1> edgelist2> ... edgelistN> \n" \
                                             "log level in steps of 10, debug=10, info=20, warning=30"
        sys.exit(-1)

    #passed cli arguments start at 1
    log_level = int(sys.argv[1])
    logging.basicConfig(level=log_level)
    snapshots = sys.argv[2:]

    np.set_printoptions(precision=3, linewidth=200)

    graphs = []
    sses = []

    #Read graphs to memory
    for snapshot in snapshots:
        logging.info("Loading graph from file: " + snapshot)
        graphs.append(Graph.Read_Ncol(snapshot, directed=False))
        #initialize size of sses list
        sses.append(([], []))

    #Perform evolutionary clustering for k=2 until 10
    for k in xrange(2, 10 + 1):
        #for each k we need to reset time and prev membership matrix
        t = 0
        prev_membership_matrix = None
        for graph in graphs:
            sse = 0
            #we cluster for the first time
            if prev_membership_matrix == None:
                (prev_membership_matrix, sse) = perform_spectral_clustering_onGraph(graph, k)
            else:
                (prev_membership_matrix, sse) = perform_evolutionary_clustering(graph, prev_membership_matrix, 0.5, k)

            #store sse value as a function of k for later plotting
            #python lists preserve input order
            sses[t][0].append(k)
            sses[t][1].append(sse)
            t += 1

    pprint("Determined sum of squared errors as function of k over time")
    pprint(sses)
    for t in xrange(len(sses)):
        plt.figure()
        plt.plot(sses[t][0], sses[t][1])
        plt.title("t=" + str(t))
        plt.ylabel('SSE')
        plt.xlabel('k')
    plt.show()

    """
    #graph with ground truth used for testing
    graph = Graph.Adjacency([
                 [0,1,1,0,0,0,0,0,0],
                 [1,0,1,0,0,0,0,0,0],
                 [1,1,0,1,1,0,0,0,0],
                 [0,0,1,0,1,1,1,0,0],
                 [0,0,1,1,0,1,1,0,0],
                 [0,0,0,1,1,0,1,1,0],
                 [0,0,0,1,1,1,0,1,0],
                 [0,0,0,0,0,1,1,0,1],
                 [0,0,0,0,0,0,0,1,0]], mode='UNDIRECTED')
    #plot(graph)
    perform_spectral_clustering_onGraph(graph,2)
    """
from collections import defaultdict
from math import inf
import random
import csv
import numpy as np

def point_avg(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    (points can have more dimensions than 2)
    
    Returns a new point which is the center of all the points.
    """
    return np.mean(points,axis=1)


def update_centers(dataset, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes 
    of both lists correspond to each other.
    Compute the center for each of the assigned groups.
    Return `k` centers in a list
    """
    cluster = np.unique(assignments)
    centroids=[]
    for i in range(len(assignments)):
        cluster = cluster[i]
        data=[]
        for j in range(len(dataset)):
            if assignments[i]==cluster:
                data.append(dataset[j])
        data=np.array(data)
        centroid = point_avg(data)

        centroids.append(centroid)
    return centroids



def assign_points(data_points, centers):
    """
    """
    assignments = []
    for point in data_points:
        shortest = inf  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    Returns the Euclidean distance between a and b
    """
    return np.linalg.norm(a-b)

def distance_squared(a, b):
    return distance(a,b)**2

def generate_k(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    """
    return np.random.choice(dataset,k, replace=0)


def cost_function(clustering):
    clusters= list(clustering)
    cost=0
    for i in range(len(clusters)):
        cluster=clusters[i]
        data=clustering[cluster]
        for j in range(len(data)):
            for k in range(len(data)):
                cost += distance(data[j],data[j])
    return cost


def generate_k_pp(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    where points are picked with a probability proportional
    to their distance as per kmeans pp
    """
    initial_centroids=generate_k(dataset,k)

    probs=[]
    for j in range(len(dataset)):
        for i in range(len(initial_centroids)):
            centroid= initial_centroids[i]
            cost +=distance_squared(dataset[j],centroid)

        probs.append(cost)
    k_clusters=[]
    for i in range(len(probs)):
        index = np.argmax(probs)
        k_clusters.append(dataset[index])
        probs[index]=np.min(probs)

    return np.array(k_clusters)


def _do_lloyds_algo(dataset, k_points):
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    clustering = defaultdict(list)
    for assignment, point in zip(assignments, dataset):
        clustering[assignment].append(point)
    return clustering


def k_means(dataset, k):
    if k not in range(1, len(dataset)+1):
        raise ValueError("lengths must be in [1, len(dataset)]")
    
    k_points = generate_k(dataset, k)
    return _do_lloyds_algo(dataset, k_points)


def k_means_pp(dataset, k):
    if k not in range(1, len(dataset)+1):
        raise ValueError("lengths must be in [1, len(dataset)]")

    k_points = generate_k_pp(dataset, k)
    return _do_lloyds_algo(dataset, k_points)

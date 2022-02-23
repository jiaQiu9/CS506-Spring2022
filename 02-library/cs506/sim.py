from numpy import dot
from numpy.linalg import norm

def euclidean_dist(x, y):
    res = 0
    for i in range(len(x)):
        res += (x[i] - y[i])**2
    return res**(1/2)

def manhattan_dist(x, y):
    res =0
    for i in range(len(x)):
        res+= (x[i] - y[i])
    return res

def jaccard_dist(x, y):
    #Find symmetric difference of two sets
    nominator = x.symmetric_difference(y)

    #Find union of two sets
    denominator = x.union(y)

    #Take the ratio of sizes
    distance = len(nominator)/len(denominator)
    
    return distance

def cosine_sim(x, y):
    #return 0
    return dot(x, y)/(norm(x)*norm(y))

# Feel free to add more

from sklearn.cluster import cluster_optics_dbscan
from sklearn.metrics import euclidean_distances
from .sim import euclidean_distances

import numpy as np
import matplotlib.pyplot as plt

class DBC():

    def __init__(self, dataset, min_pts, epsilon):
        self.dataset = dataset
        self.min_pts = min_pts
        self.epsilon = epsilon

    def epsilon_neighborhood(self, P):
        # TODO: implement next time
        return []


    def explore_and_assign_eps_neighborhood(self, P, cluster, assignments):
        # TODO: implement next time
        return assignments


    def snapshot(self, P,assignment):
        fig, ax= plt.subplots()
        colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
        colors = np.hstack([colors] * 20)
        ax.scatter(self.dataset[:, 0], self.dataset[:, 1], color=colors[assignment].tolist(), s=10, alpha=0.8)
        cir=plt.Circle((self.dataset[P][0],self.dataset[P][1]), )
        fig.savefig("temp.png")
        plt.close()

    def epsilon_neighbourhood(self, P_index):
        neighborhood=[]
        for PN in range(len(self.dataset)):
            if P_index!=PN and euclidean_distances(self.dataset[PN], self.dataset[P_index]) <= self.epsilon:
                #in the neighborhood

                neighborhood.append(PN) 

        return []

    def explore_and_assign_eps_neighbourhood(self, P_index, cluster, assignments):

        neighborhood= self.epsilon_neighbourhood(P_index)


        while neighborhood:
            neighbor_of_P= neighborhood.pop()


            if assignments[neighbor_of_P]!=0:
                #this point has already been assigned
                # no point in assigning it again
                continue 

            assignments[neighbor_of_P]=cluster 
            self.snapshot(neighbor_of_P, assignments) # take a snapshot after assingments. 

            next_neighborhood=self.epsilon_neighbourhood(neighbor_of_P)

            if len(next_neighborhood)>= self.min_pts:
                # this is a core point 
                # its neighbors should be explored/ assigned also
                neighborhood.extend(next_neighborhood)

        return assignments


    def dbscan(self):
        """
            returns a list of assignments. The index of the
            assignment should match the index of the data point
            in the dataset.
        """

        assignments = [0 for _ in range(len(self.dataset))]
        cluster = 1

        for P in range(len(self.dataset)):
            
            if assignments[P] != 0:
                # already part of a cluster
                continue

            if len(self.epsilon_neighborhood(P)) >= self.min_pts:
                # core point
                assignments = self.explore_and_assign_eps_neighborhood(
                    P, cluster, assignments)

            cluster += 1

        return assignments

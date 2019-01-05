import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def semanticGraph(data,labels):

    adjacency = cosine_similarity(labels,labels)

    np.fill_diagonal(adjacency,0)

    return adjacency



def structureGraph(data,k=5):
    n=len(data)
    adjacency = np.zeros((n,n))

    distances = euclidean_distances(data,data)

    for i in range(n):
        k_nearest_neighbors = np.argsort(distances[i,:],kind='heapsort')[1:k+1]
        for j in k_nearest_neighbors:
            adjacency[i,j] = 1
#        adjacency[i,:] = [1 if j in k_nearest_neighbors else 0 for j in range(n)]

    adjacency = np.maximum(adjacency,adjacency.T)

    return adjacency



def sampleStructureNeighborhood(adjacency,k):
    n = len(adjacency)
    neighbors = []
    for i in range(n):
        multinomial = adjacency[i,:]/np.sum(adjacency[i,:])
        occurences = np.random.multinomial(k,multinomial)
        neighborhood = []
        for node in range(n):
            neighborhood += [node]*occurences[node]
        neighbors.append(neighborhood)

    return neighbors

def sampleSemanticNeighborhood(adjacency,k,indexes_sample,indexes_neighbors):
    adjacency = adjacency[indexes_sample,:][:,indexes_neighbors]
    n = len(adjacency)
    neighbors = []
    for i in range(n):
        multinomial = adjacency[i,:]/np.sum(adjacency[i,:])
        occurences = np.random.multinomial(k,multinomial)
        neighborhood = []
        for node in range(n):
            neighborhood += [node]*occurences[node]
        neighbors.append(neighborhood)

    return neighbors

def sampleSemanticNeighborhoodSame(adjacency,k,indexes_sample):
    indexes_neighbors = [i for i in range(len(adjacency)) if i in indexes_sample]
    return sampleSemanticNeighborhood(adjacency,k,indexes_sample,indexes_neighbors)


def sampleSemanticNeighborhoodOther(adjacency,k,indexes_sample):
    indexes_neighbors = [i for i in range(len(adjacency)) if i not in indexes_sample]
    return sampleSemanticNeighborhood(adjacency,k,indexes_sample,indexes_neighbors)


def sampleStructureNegative(adjacency,k):
    n = len(adjacency)
    neighbors = []
    for i in range(n):
        disconnected = [j for j in range(n) if adjacency[i,j] == 0.]
        neighbors.append([np.random.choice(disconnected) for _ in range(k)])

    return neighbors

def sampleSemanticNegative(adjacency,k,indexes_sample,indexes_neighbors):
    adjacency = adjacency[indexes_sample,:][:,indexes_neighbors]
    n = len(adjacency)
    neighbors = []
    for i in range(n):
        disconnected = [j for j in range(n) if adjacency[i,j] == 0.]
        neighbors.append([np.random.choice(disconnected) for _ in range(k)])

    return neighbors

def sampleSemanticNegativeSame(adjacency,k,indexes_sample):
    indexes_neighbors = [i for i in range(len(adjacency)) if i in indexes_sample]
    return sampleSemanticNegative(adjacency,k,indexes_sample,indexes_neighbors)


def sampleSemanticNegativeOther(adjacency,k,indexes_sample):
    indexes_neighbors = [i for i in range(len(adjacency)) if i not in indexes_sample]
    return sampleSemanticNegative(adjacency,k,indexes_sample,indexes_neighbors)

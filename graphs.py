import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def semanticGraph(data,labels):
    n=len(data)
    adjacency = np.zeros((n,n))
    labels = np.array(labels)
    for i in range(n):
        adjacency[i:,i] = cosine_similarity([labels[i]],labels[i:])
        adjacency[i,i:] = adjacency[i:,i]
        adjacency[i,i] = 0

    return adjacency



def structureGraph(data,k=5):
    n=len(data)
    adjacency = np.zeros((n,n))

    distances = [[np.linalg.norm(v1-v2) for v1 in data] for v2 in data]

    for i in range(n):
        k_nearest_neighbors = np.argsort(distances[i])[-k-1:-1]
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
        neighbors.append([np.random.choice(disconnected)])

    return neighbors

def sampleSemanticNegative(adjacency,k,indexes_sample,indexes_neighbors):
    adjacency = adjacency[indexes_sample,:][:,indexes_neighbors]
    n = len(adjacency)
    neighbors = []
    for i in range(n):
        disconnected = [j for j in range(n) if adjacency[i,j] == 0.]
        neighbors.append([np.random.choice(disconnected)])

    return neighbors

def sampleSemanticNegativeSame(adjacency,k,indexes_sample):
    indexes_neighbors = [i for i in range(len(adjacency)) if i in indexes_sample]
    return sampleSemanticNegative(adjacency,k,indexes_sample,indexes_neighbors)


def sampleSemanticNegativeOther(adjacency,k,indexes_sample):
    indexes_neighbors = [i for i in range(len(adjacency)) if i not in indexes_sample]
    return sampleSemanticNegative(adjacency,k,indexes_sample,indexes_neighbors)

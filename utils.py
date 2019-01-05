import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse

def getParsedArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--structure_context_size',default=8,help='Numbers of sampled neighbors in structural graphs.',type=int)
    parser.add_argument('--structure_negative_context_size',default=8,help='Numbers of negative sampled neighbors in structural graphs.',type=int)

    parser.add_argument('--same_semantic_context_size',default=2,help='Numbers of sampled neighbors in semantic graphs of the same modality.',type=int)
    parser.add_argument('--other_semantic_context_size',default=6,help='Numbers of sampled neighbors in semantic graphs of the other modality.',type=int)
    parser.add_argument('--same_semantic_negative_context_size',default=2,help='Numbers of sampled negative neighbors in semantic graphs of the same modality.',type=int)
    parser.add_argument('--other_semantic_negative_context_size',default=6,help='Numbers of sampled negative neighbors in semantic graphs of the other modality.',type=int)

    parser.add_argument('--k_nearest',default=8,help='Numbers of nearest neighbors for structural graph construction.',type=int)

    parser.add_argument('--n_epochs',default=8,help='Number of epochs for training on the generated dataset.',type=int)
    parser.add_argument('--batch_size',default=5,help='Numbers of images and texts in batches used for mini batch training.',type=int)
    parser.add_argument('--starter_learning_rate',default=0.0001,help='Weight of the semantic graph loss.',type=float)

    parser.add_argument('--alpha_global',default=0.1,help='Weight of the global semantic loss.',type=float)
    parser.add_argument('--alpha_structure',default=0.1,help='Weight of the structural graph loss.',type=float)
    parser.add_argument('--alpha_semantic',default=1.,help='Weight of the semantic graph loss.',type=float)

    return parser.parse_args()

def tf_repeat(tensor,n_repeats):
    return tf.concat(tf.unstack(tf.keras.backend.repeat(tensor,n_repeats),axis=0),axis=0)

def matchingLabels(query_labels,database_labels):
    return [True for label1,label2 in zip(query_labels,database_labels) if (label1) and (label2)]

def meanAveragePrecision(query_set,database_set,query_labels,database_labels,R):
    similarities = cosine_similarity(query_set,database_set)

    hits = np.zeros((similarities.shape[0],R))

    for i in range(len(similarities)):
        top_R_matches=np.argsort(-similarities[i,:])[:R]
        for j,r in enumerate(top_R_matches):
            if matchingLabels(query_labels[i],database_labels[r]):
                hits[i,j] = 1
    #return hits
    average_precisions = []
    for i in range(len(similarities)):
        precisions =[]
        for r in range(R):
            if hits[i,r]:
                precisions.append(np.mean(hits[i,:r+1]))
        average_precisions.append(np.mean(precisions))

    return np.mean(average_precisions)

def recallAtK(query_set,database_set,query_labels,database_labels,K):
    similarities = cosine_similarity(query_set,database_set)

    hits = np.zeros((similarities.shape[0]))

    for i in range(len(similarities)):
        top_R_matches=np.argsort(-similarities[i,:])[:K]
        for j,r in enumerate(top_R_matches):
            if matchingLabels(query_labels[i],database_labels[r]):
                hits[i] = 1
                break
    #return hits
    average_precisions = []

    return np.mean(hits)

def medR(query_set,database_set,query_labels,database_labels,K):
    similarities = cosine_similarity(query_set,database_set)

    hits = np.zeros((similarities.shape[0]))

    for i in range(len(similarities)):
        top_R_matches=np.argsort(-similarities[i,:])[:K]
        for j,r in enumerate(top_R_matches):
            if matchingLabels(query_labels[i],database_labels[r]):
                hits[i] = j
                break
    #return hits

    return np.median(hits)+1

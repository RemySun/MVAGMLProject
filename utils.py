import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse

def getParsedArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',default='5',help='Numbers of images and texts in batches used for mini batch training.')
    return parser.parse_args()

def tf_repeat(tensor,n_repeats):
    return tf.concat(tf.unstack(tf.keras.backend.repeat(tensor,n_repeats),axis=0),axis=0)

def matchingLabels(query_labels,database_labels):
    return [True for label1,label2 in zip(query_labels,database_labels) if (label1) and (label2)]

def meanAveragePrecision(query_set,database_set,query_labels,database_labels,R):
    similarities = cosine_similarity(query_set,database_set)

    hits = np.zeros(similarities.shape)

    for i in range(len(similarities)):
        top_R_matches=np.argsort(similarities[i,:])[-R:]
        for r in top_R_matches:
            if matchingLabels(query_labels[i],database_labels[r]):
                hits[i,r] = 1

    average_precisions = []
    for i in range(len(similarities)):
        precisions =[]
        for r in range(R):
            if hits[i,r]:
                precisions.append(np.mean(hits[i,:r+1]))
        average_precisions.append(np.mean(precisions))

    return np.mean(average_precisions)


    

import tensorflow as tf
import numpy as np
import graphs
import data
import networks
import utils
import time
from sklearn.metrics.pairwise import cosine_similarity

#################################
#################################
#####                       #####
#####    Hyperparameters    #####
#####                       #####
#################################
#################################

k_nearest_imgs = 100
k_nearest_text = 100

nb_iterations = 20

batch_size = 10

alpha_global=0.1
alpha_structure=0.1

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.01
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           2000, 0.95, staircase=True)


########################################
########################################
#####                              #####
#####    Load & preprocess data    #####
#####                              #####
########################################
########################################

print('Loading processed data...')
start = time.time()

data_imgs = data.load_imgs('img_features.p')
data_text = data.load_text('pascal/train.mat')
labels_text = labels_imgs = data.load_labels('VOCdevkit/VOC2007/ImageSets/Main',5011,20)

IMG_SIZE = len(data_imgs[0])
TEXT_SIZE = len(data_text[0])
LABEL_SIZE = len(labels_text[0])

data = list(data_imgs)+ data_text
labels = list(labels_imgs)+ list(labels_text)

idx_imgs = [i for i in range(len(data_imgs))]
idx_text = [len(data_imgs)+i for i in range(len(data_text))]

print("Finished loading preprocessed data, it took ", time.time()-start)
start = time.time()

###################################
###################################
#####                         #####
#####    Create loss graph    #####
#####                         #####
###################################
###################################

print("Creating the network and loss graphs...")
start = time.time()

imgs_placeholder = tf.placeholder(tf.float32,shape=[None,IMG_SIZE])
text_placeholder = tf.placeholder(tf.float32,shape=[None,TEXT_SIZE])

imgs_encoded = networks.make_encoder(imgs_placeholder,name='imgs_encoder',n_input=IMG_SIZE)
text_encoded = networks.make_encoder(text_placeholder,name='text_encoder',n_input=TEXT_SIZE)

saver = tf.train.Saver()

sess = tf.Session()

saver.restore(sess,"checkpoints/model.ckpt")

imgs_latent, text_latent = sess.run([imgs_encoded,text_encoded],feed_dict={imgs_placeholder:data_imgs,text_placeholder:data_text})

similarities = cosine_similarities([imgs_latent,text_latent])

hits = 0
K = 5
for i in similarities:
    top_K_matches=np.argsort(similarities[i,:])[-K:]
    for k in top_K_matches:
        if len([True for label1,label2 in zip(labels_imgs[i],labels_text[k]) if label1==label2]):
            hits += 1

precision = hits/(len(similarities)*K)

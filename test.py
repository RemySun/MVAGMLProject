import tensorflow as tf
import numpy as np
import graphs
import data
import networks
import utils
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

args = utils.getParsedArgs()

########################################
########################################
#####                              #####
#####    Load & preprocess data    #####
#####                              #####
########################################
########################################
print('new and improved testing')
print('Loading processed data...')
start = time.time()

data_imgs = data.load_imgs('VOC_alexnet_feat_test.p')
data_text = np.array(data.load_text('pascal/test.mat'))
labels_text = labels_imgs = data.load_labels_test('VOCTest/VOCdevkit/VOC2007/ImageSets/Main',4952,20)

# data_imgs = data.load_imgs('img_features.p')
# data_text = np.array(data.load_text('pascal/train.mat'))
# labels_text = labels_imgs = data.load_labels('VOCdevkit/VOC2007/ImageSets/Main',5011,20)


valid_idx = (np.sum(data_text,axis=1) != np.zeros(data_text.shape[0]))

data_imgs=preprocessing.scale(data_imgs[valid_idx])
data_text=preprocessing.scale(data_text[valid_idx])
labels_text=labels_imgs=labels_text[valid_idx]

IMG_SIZE = len(data_imgs[0])
TEXT_SIZE = len(data_text[0])
LABEL_SIZE = len(labels_text[0])

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

similarities = cosine_similarity(imgs_latent,text_latent)

# hits = 0
# K = 200
# for i in range(len(similarities)):
#     top_K_matches=np.argsort(-similarities[i,:])[:K]
#     for k in top_K_matches:
#         if len([True for label1,label2 in zip(labels_imgs[i],labels_text[k]) if (label1) and (label2)]):
#             hits += 1

# precision_it = hits/(len(similarities)*K)

# similarities = cosine_similarity(text_latent,imgs_latent)

# hits = 0
# K = 200
# for i in range(len(similarities)):
#     top_K_matches=np.argsort(-similarities[i,:])[:K]
#     for k in top_K_matches:
#         if len([True for label1,label2 in zip(labels_text[i],labels_imgs[k]) if (label1) and (label2)]):
#             hits += 1

# precision_ti = hits/(len(similarities)*K)

#hits_mat =(utils.meanAveragePrecision(imgs_latent,text_latent,labels_imgs,labels_text,200))
print(utils.meanAveragePrecision(imgs_latent,text_latent,labels_imgs,labels_text,len(imgs_latent)))
print(utils.meanAveragePrecision(text_latent,imgs_latent,labels_text,labels_imgs,len(imgs_latent)))

for k in [1,5,10]:
    print('Recall at',k)
    print(utils.recallAtK(imgs_latent,text_latent,labels_imgs,labels_text,k))
    print(utils.recallAtK(text_latent,imgs_latent,labels_text,labels_imgs,k))

print('median rank is')
print(utils.medR(text_latent,imgs_latent,labels_text,labels_imgs,500))


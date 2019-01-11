import tensorflow as tf
import numpy as np
import graphs
import data
import networks
import utils
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
import scipy.io as sio
import glob
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

data_imgs = data.load_imgs(args.imgs_feat_path)
data_text = np.array(data.load_text('pascal/train.mat'))

valid_idx = (np.sum(data_text,axis=1) != np.zeros(data_text.shape[0]))

imgs_scaler = preprocessing.StandardScaler()
text_scaler = preprocessing.StandardScaler()

data_imgs=imgs_scaler.fit_transform(data_imgs[valid_idx])
data_text=text_scaler.fit_transform(data_text[valid_idx])

data_imgs = data.load_imgs(args.imgs_feat_test_path)
data_text = np.array(data.load_text('pascal/test.mat'))
labels_text = labels_imgs = data.load_labels_test('VOCTest/VOCdevkit/VOC2007/ImageSets/Main',4952,20)

valid_idx = (np.sum(data_text,axis=1) != np.zeros(data_text.shape[0]))

vanilla_text = data_text[valid_idx]
vanilla_imgs = data_imgs[valid_idx]

data_imgs=imgs_scaler.transform(data_imgs[valid_idx])
data_text=text_scaler.transform(data_text[valid_idx])
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

# sample_names = {}
# for i,filename in enumerate(np.sort(glob.glob("VOCdevkit/VOC2007/JPEGImages/train/*"))):
#     sample_names.update({i:int(filename.split('/')[-1].split('.')[0])})


voc=sio.loadmat('pascal/voc')

voc = dict(zip([i for i in range(399)],[voc['voc'][i][0][0] for i in range(399)]))

# sim_imgs_text = cosine_similarity(imgs_latent,text_latent)
# sim_text_imgs = cosine_similarity(text_latent,imgs_latent)

# text_hits = np.argsort(sim_imgs_text[0,:])[:3]
# imgs_hits = np.argsort(sim_text_imgs[0,:])[:3]

# for text_hit in text_hits:
#     print('one_hit')
#     for i,tag in enumerate(vanilla_text[text_hit]):
#         if tag !=0:
#             print(voc[i])

# print('text_query')
# for i,tag in enumerate(vanilla_text[0]):
#     if tag !=0:
#         print(voc[i])
# img_names = np.sort(glob.glob('VOCTest/VOCdevkit/VOC2007/JPEGImages/*.jpg'))[valid_idx]

# for img_hit in imgs_hits:
#     print('one_hit')
#     for i in (img_names[img_hit]):
#         print(i)


print(utils.meanAveragePrecision(imgs_latent,text_latent,labels_imgs,labels_text,len(imgs_latent)))
print(utils.meanAveragePrecision(text_latent,imgs_latent,labels_text,labels_imgs,len(imgs_latent)))

for k in [1,5,10]:
    print('Recall at',k)
    print(utils.recallAtK(imgs_latent,text_latent,labels_imgs,labels_text,k))
    print(utils.recallAtK(text_latent,imgs_latent,labels_text,labels_imgs,k))

print('median rank is')
print(utils.medR(text_latent,imgs_latent,labels_text,labels_imgs,500))

print('Image to text')
print(utils.comprehensiveEval(imgs_latent,text_latent,labels_imgs,labels_text))
print('Text to image')
print(utils.comprehensiveEval(text_latent,imgs_latent,labels_text,labels_imgs))


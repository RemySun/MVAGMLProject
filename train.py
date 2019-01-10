import tensorflow as tf
import numpy as np
import graphs
import data
import networks
import utils
import time
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances

args = utils.getParsedArgs()

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(args.starter_learning_rate, global_step,
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

data_imgs = data.load_imgs(args.imgs_feat_path)
data_text = np.array(data.load_text('pascal/train.mat'))
labels_text = labels_imgs = data.load_labels('VOCdevkit/VOC2007/ImageSets/Main',5011,20)

valid_idx = (np.sum(data_text,axis=1) != np.zeros(data_text.shape[0]))

imgs_scaler = preprocessing.StandardScaler()
text_scaler = preprocessing.StandardScaler()

data_imgs=imgs_scaler.fit_transform(data_imgs[valid_idx])
data_text=text_scaler.fit_transform(data_text[valid_idx])
labels_text=labels_imgs=labels_text[valid_idx]

IMG_SIZE = len(data_imgs[0])
TEXT_SIZE = len(data_text[0])
LABEL_SIZE = len(labels_text[0])

data = list(data_imgs)+ list(data_text)
labels = list(labels_imgs)+ list(labels_text)

idx_imgs = [i for i in range(len(data_imgs))]
idx_text = [len(data_imgs)+i for i in range(len(data_text))]

print("Finished loading preprocessed data, it took ", time.time()-start)
start = time.time()

adjacency_semantic = graphs.semanticGraph(data, labels)

adjacency_structure_imgs = graphs.structureGraph(data_imgs, k=args.k_nearest)
adjacency_structure_text = graphs.structureGraph(data_text, k=args.k_nearest)

print("Finished building data graphs, it took ", time.time()-start)

###################################
###################################
#####                         #####
#####    Create loss graph    #####
#####                         #####
###################################
###################################

print("Creating the network and loss graphs...")
start = time.time()

imgs_placeholder = tf.placeholder(tf.float32,shape=[args.batch_size,IMG_SIZE])
text_placeholder = tf.placeholder(tf.float32,shape=[args.batch_size,TEXT_SIZE])

imgs_encoded = networks.make_encoder(imgs_placeholder,name='imgs_encoder',n_input=IMG_SIZE)
text_encoded = networks.make_encoder(text_placeholder,name='text_encoder',n_input=TEXT_SIZE)

####################################
### Compute global semantic loss ###
####################################

imgs_labels_placeholder = tf.placeholder(tf.float32,shape=[None,LABEL_SIZE])
text_labels_placeholder = tf.placeholder(tf.float32,shape=[None,LABEL_SIZE])

imgs_decoded = (networks.make_decoder(imgs_encoded))
text_decoded = (networks.make_decoder(text_encoded,reuse=True))

global_semantic_loss = (tf.reduce_sum(tf.square(imgs_decoded-imgs_labels_placeholder))
                        + tf.reduce_sum(tf.square(text_decoded-text_labels_placeholder)))

######################################
### Compute graph embedding losses ###
######################################

# First dim: input (imgs,text,structure)
# Second_dim: context (semantic same mod, semantic other_mod, structure (same mod)) (repeated for negative samples)
context_placeholders = [
    [
        tf.placeholder(tf.float32,shape=[None,IMG_SIZE]),
        tf.placeholder(tf.float32,shape=[None,TEXT_SIZE]),
        tf.placeholder(tf.float32,shape=[None,IMG_SIZE])
    ],
    [
        tf.placeholder(tf.float32,shape=[None,TEXT_SIZE]),
        tf.placeholder(tf.float32,shape=[None,IMG_SIZE]),
        tf.placeholder(tf.float32,shape=[None,TEXT_SIZE])
    ],
    [
        tf.placeholder(tf.float32,shape=[None,IMG_SIZE]),
        tf.placeholder(tf.float32,shape=[None,TEXT_SIZE]),
        tf.placeholder(tf.float32,shape=[None,IMG_SIZE])
    ],
    [
        tf.placeholder(tf.float32,shape=[None,TEXT_SIZE]),
        tf.placeholder(tf.float32,shape=[None,IMG_SIZE]),
        tf.placeholder(tf.float32,shape=[None,TEXT_SIZE])
    ]
]

encoder_names = np.tile([
    ['imgs_encoder','text_encoder','imgs_encoder'],
    ['text_encoder','imgs_encoder','text_encoder']
],[2,1])

context_encodings = [
    [
        networks.make_encoder(context_placeholders[i][j],name=encoder_names[i][j],reuse=True,n_input=int(context_placeholders[i][j].shape[1])) for j in range(3)
    ] for i in range(4)
]

context_sizes = [
    [args.same_semantic_context_size,args.other_semantic_context_size,args.structure_context_size],
    [args.same_semantic_negative_context_size,args.other_semantic_negative_context_size,args.structure_negative_context_size]
]

input_encodings = [
    networks.make_encoder(imgs_placeholder,name='imgs_encoder',reuse=True,n_input=IMG_SIZE),
    networks.make_encoder(text_placeholder,name='text_encoder',reuse=True,n_input=TEXT_SIZE)
]

context_signs = [1,-1]
context_alphas = [args.alpha_semantic,args.alpha_semantic,args.alpha_structure]
losses_skip = [
    [
       -context_alphas[j]*tf.reduce_sum(tf.log(tf.sigmoid(context_signs[i//2]*tf.reduce_sum(tf.multiply(utils.tf_repeat(input_encodings[i%2],context_sizes[i//2][j]),context_encodings[i][j]),axis=1)))) for j in range(3)
    ] for i in range(4)
]

loss_skip = tf.zeros([1])
for losses in losses_skip:
    for loss in losses:
        loss_skip += loss


loss = args.alpha_global * global_semantic_loss + loss_skip

weights = tf.trainable_variables()[1:11]
weight_decay_l2 = tf.zeros([1])
for w in weights:
    weight_decay_l2 += tf.reduce_sum(tf.square(w))
weight_decay_op = tf.train.GradientDescentOptimizer(0.00005).minimize(weight_decay_l2)

update_loss = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step=global_step)

print("Finished building the Tensorflow graph, it took", time.time()-start)

#####################################
#####################################
#####                           #####
#####    Training iterations    #####
#####                           #####
#####################################
#####################################

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

sess = tf.Session()

sess.run(init_op)

print("Starting training...")

order_imgs = np.array([i for i in range(len(data_imgs)*5)])
order_text = np.array([i for i in range(len(data_text)*5)])

print('Sampling neighborhoods')
start=time.time()
if args.use_walks:
    contexts_tot = np.array([[
        [
            graphs.sampleSemanticWalkSame(adjacency_semantic,args.same_semantic_context_size,idx_imgs),
            graphs.sampleSemanticWalkOther(adjacency_semantic,args.other_semantic_context_size,idx_imgs),
            graphs.sampleStructureWalk(adjacency_structure_imgs,args.structure_context_size)
        ],
        [
            graphs.sampleSemanticWalkSame(adjacency_semantic,args.same_semantic_context_size,idx_text),
            graphs.sampleSemanticWalkOther(adjacency_semantic,args.other_semantic_context_size,idx_text),
            graphs.sampleStructureWalk(adjacency_structure_text,args.structure_context_size)
        ],
        [
            graphs.sampleSemanticNegativeSame(adjacency_semantic,args.same_semantic_negative_context_size,idx_imgs),
            graphs.sampleSemanticNegativeOther(adjacency_semantic,args.other_semantic_negative_context_size,idx_imgs),
            graphs.sampleStructureNegative(adjacency_structure_imgs,args.structure_negative_context_size)
        ],
        [
            graphs.sampleSemanticNegativeSame(adjacency_semantic,args.same_semantic_negative_context_size,idx_text),
            graphs.sampleSemanticNegativeOther(adjacency_semantic,args.other_semantic_negative_context_size,idx_text),
            graphs.sampleStructureNegative(adjacency_structure_text,args.structure_negative_context_size)
        ]
    ] for _ in range(5)]
    )
else:
    contexts_tot = np.array([[
        [
            graphs.sampleSemanticNeighborhoodSame(adjacency_semantic,args.same_semantic_context_size,idx_imgs),
            graphs.sampleSemanticNeighborhoodOther(adjacency_semantic,args.other_semantic_context_size,idx_imgs),
            graphs.sampleStructureNeighborhood(adjacency_structure_imgs,args.structure_context_size)
        ],
        [
            graphs.sampleSemanticNeighborhoodSame(adjacency_semantic,args.same_semantic_context_size,idx_text),
            graphs.sampleSemanticNeighborhoodOther(adjacency_semantic,args.other_semantic_context_size,idx_text),
            graphs.sampleStructureNeighborhood(adjacency_structure_text,args.structure_context_size)
        ],
        [
            graphs.sampleSemanticNegativeSame(adjacency_semantic,args.same_semantic_negative_context_size,idx_imgs),
            graphs.sampleSemanticNegativeOther(adjacency_semantic,args.other_semantic_negative_context_size,idx_imgs),
            graphs.sampleStructureNegative(adjacency_structure_imgs,args.structure_negative_context_size)
        ],
        [
            graphs.sampleSemanticNegativeSame(adjacency_semantic,args.same_semantic_negative_context_size,idx_text),
            graphs.sampleSemanticNegativeOther(adjacency_semantic,args.other_semantic_negative_context_size,idx_text),
            graphs.sampleStructureNegative(adjacency_structure_text,args.structure_negative_context_size)
        ]
    ] for _ in range(5)]
    )

print('Finished sampling neighborhoods, it took',time.time()-start)

contexts=np.concatenate(contexts_tot,axis=2)

for epoch in range(args.n_epochs):

    print("Starting epoch",epoch)

    epoch_start = time.time()

    np.random.shuffle(order_imgs)
    np.random.shuffle(order_text)

    data_imgs_iter = data_imgs[order_imgs%len(data_imgs)]
    data_text_iter = data_text[order_text%len(data_text)]
    labels_imgs_iter = labels_imgs[order_imgs%len(data_imgs)]
    labels_text_iter = labels_text[order_text%len(data_text)]

    orders_cont =np.array(
            [order_imgs,order_text]
        )

    contexts_iter = [[contexts[i,j][orders_cont[i%2]] for j in range(3)] for i in range(4)]


    recorded_losses =[]
    batch_start = time.time()
    for batch in range((len(data_imgs)*5)//args.batch_size):


        ### Build the dictionary fed as input to the network

        input_dict = {
            imgs_placeholder:data_imgs_iter[batch*args.batch_size:(batch+1)*args.batch_size],
            text_placeholder:data_text_iter[batch*args.batch_size:(batch+1)*args.batch_size],
            imgs_labels_placeholder:labels_imgs_iter[batch*args.batch_size:(batch+1)*args.batch_size],
            text_labels_placeholder:labels_text_iter[batch*args.batch_size:(batch+1)*args.batch_size]
        }

        orders_batch =np.array(
            [order_imgs,order_text]
        )

        contexts_batch = [[np.concatenate(contexts_iter[i][j][batch*args.batch_size:(batch+1)*args.batch_size]) for j in range(3)] for i in range(4)]
        data_context = [
            [data_imgs,data_text,data_imgs],
            [data_text,data_imgs,data_text],
            [data_imgs,data_text,data_imgs],
            [data_text,data_imgs,data_text]
        ]
        for i in range(4):
            for j in range(3):
                input_dict.update({context_placeholders[i][j]:[data_context[i][j][node] for node in contexts_batch[i][j]]})

        batch_loss, _, _ = sess.run([loss,update_loss,weight_decay_op],feed_dict = input_dict)

        recorded_losses.append(batch_loss)
        if batch % 50 == 0:
            save_path = saver.save(sess,"checkpoints/model.ckpt")
            print("Completed 50 batches in",time.time()-batch_start,"seconds with an average loss of",np.mean(recorded_losses))
            batch_start = time.time()
            recorded_losses=[]
            print("Model saved in",save_path)


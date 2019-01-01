import tensorflow as tf

def make_encoder(x,n_input=100,n_hidden_1=64,n_latent=32,name='text',reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        weights = {
            'h1': tf.get_variable('h1/weights',shape=[n_input, n_hidden_1]),

            'out': tf.get_variable('latent/weights',[n_hidden_1, n_latent])
        }
        biases = {
            'b1': tf.get_variable('h1/bias',[n_hidden_1]),

            'out': tf.get_variable('latent/bias',[n_latent])
        }


    layer_1 = tf.tanh(tf.add(tf.matmul(x, weights['h1']), biases['b1']))

    out_layer = tf.tanh(tf.add(tf.matmul(layer_1, weights['out']), biases['out']))

    return out_layer

def make_decoder(x,n_input=32,n_out=20,name='decoder',reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        weights = {
            'out': tf.get_variable('out/weights',[n_input, n_out])
        }
        biases = {
            'b_out': tf.get_variable('out/bias',[n_out])
        }


    out_layer = (tf.add(tf.matmul(x, weights['out']), biases['b_out']))

    return out_layer

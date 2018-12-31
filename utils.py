import tensorflow as tf

def tf_repeat(tensor,n_repeats):
    return tf.concat(tf.unstack(tf.keras.backend.repeat(tensor,n_repeats),axis=0),axis=0)

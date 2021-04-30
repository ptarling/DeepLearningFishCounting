import tensorflow as tf


def hinge_loss(y_true, y_pred):
    yhat_diff = y_pred[:,0]
    return tf.reduce_sum(tf.math.maximum(y_true, yhat_diff)) 


def au_absolute_loss(y_true, y_pred):
    y_hat = tf.reduce_sum((y_pred[:,:,:,0]), [1,2])
    y = tf.reduce_sum((y_true[:,:,:,0]), [1,2])
    log_var = tf.reduce_sum(y_pred[:,:,:,1], [1,2])
    adjusted_ab = tf.math.multiply(tf.math.abs(y_hat - y),tf.math.exp(-log_var))        
    return tf.reduce_sum(tf.reduce_sum((adjusted_ab, log_var),0))


def MAE(y_true, y_pred):
    y_hat = tf.reduce_sum((y_pred[:,:,:,0]), [1,2])
    y = tf.reduce_sum((y_true[:,:,:,0]), [1,2])
    return tf.reduce_mean(tf.math.abs(y - y_hat),0)


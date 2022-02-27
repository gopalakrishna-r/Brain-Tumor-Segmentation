import tensorflow.keras.backend as K
from tensorflow.keras.losses import categorical_crossentropy

def generalized_dice(y_true, y_pred):
    y_true = K.reshape(y_true, shape = (-1,4))
    y_pred = K.reshape(y_pred, shape = (-1,4))
    sum_p = K.sum(y_pred, -2)
    sum_r = K.sum(y_true, -2)
    sum_pr = K.sum(y_true * y_true, -2)
    weights = K.pow(K.square(sum_r) + K.epsilon(), -1)
    return (2 * K.sum(weights * sum_pr)) / (K.sum(weights * (sum_r + sum_p)))

def generalized_dice_loss(y_true, y_pred):
    return 1- generalized_dice(y_true, y_pred)

def custom_loss(y_true, y_pred):
    return generalized_dice(y_true, y_pred) + 1.25 * categorical_crossentropy(y_true, y_pred)
    
import numpy as np
import tensorflow as tf


def iou(y_true, y_pred):
    def IOU(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        total_iou = (intersection + 1e-15) / (union + 1e-15)
        total_iou = total_iou.astype(np.float32)
        return total_iou

    return tf.numpy_function(IOU, [y_true, y_pred], tf.float32)


def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    dc = (2 * intersection + 1e-15) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-15)
    return dc


def dice_loss(y_true, y_pred):
    loss = 1.0 - dice_coef(y_true, y_pred)
    return loss

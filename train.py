import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from glob import glob
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import cv2
from model import DeeplabV3Plus
from metrics import iou, dice_coef, dice_loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, TensorBoard


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path, shuffling=False):
    """Takes images folder path and returns train and test data"""
    img_path  = os.path.join(path, 'images', '*.png')
    mask_path = os.path.join(path, 'masks', '*.png')

    img_list  = sorted(glob(img_path))
    mask_list = sorted(glob(mask_path))

    if shuffling:
        img_list, mask_list = shuffle(img_list, mask_list, random_state=42)

    return img_list, mask_list


# def read_image(path, mask=False):
#     # path = path.decode()
#     if mask:
#         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         img = np.expand_dims(img, axis=-1)
#     else:
#         img = cv2.imread(path, cv2.IMREAD_COLOR)
#     img = img / 255.0
#     img = img.astype(np.float32)

#     return img


# @tf.function
# def parse_func(x, y):
#     img = tf.io.read_file(x)
#     print(img)
#     img = tf.image.decode_png(img, channels=3)
#     print(img)
#     img = tf.cast(img, dtype=tf.float32)
#     print(img)
#     img = img.numpy() / 255.0
    

#     mask = tf.io.read_file(y)
#     mask = tf.image.decode_png(y, channels=1)
#     # tf.cast(mask, dtype=tf.float32)
#     # mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
#     mask.set_shape([None, None, 1])
#     mask = mask.numpy() / 255.0
    

#     return img, mask
    

# def tf_dataset(x, y, batch_size=2):
#     dataset = tf.data.Dataset.from_tensor_slices((x,y))
#     dataset = dataset.map(parse_func, num_parallel_calls=tf.data.AUTOTUNE)
#     dataset = dataset.batch(batch_size)
#     dataset = dataset.prefetch(tf.data.AUTOTUNE)
#     return dataset

def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = image / 255.0
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = image / 255.0
    return image


def load_images(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask


def tf_dataset(image_list, mask_list, batch_size=2):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)
    BATCH_SIZE = 2
    num_epochs = 20
    IMAGE_SIZE = 512
    NUM_CLASSES = 1
    lr = 1e-4

    create_dir('model_output')

    model_path = os.path.join('model_output', 'model.h5')
    csv_path = os.path.join('model_output', 'train_history.csv')

    data_path = 'people_segmentation/augumented_data'
    train_path = os.path.join(data_path, "train")
    valid_path = os.path.join(data_path, "test")

    train_img, train_mask = load_data(train_path, shuffling=True)
    valid_img, valid_mask = load_data(valid_path, shuffling=False)

    print(f'number of train images : {len(train_img)}, number of train masks : {len(train_mask)}')
    print(f'number of valid images : {len(valid_img)} , number of test masks : {len(valid_mask)}')

    train_dataset = tf_dataset(train_img, train_mask, batch_size=BATCH_SIZE)
    valid_dataset = tf_dataset(valid_img, valid_mask, batch_size=BATCH_SIZE)

    model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision()])

    callbacks = [
        ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=1e-7),
        CSVLogger(csv_path),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=False)
    ]

    model.fit(
        train_dataset, 
        epochs= num_epochs, 
        validation_data=valid_dataset, 
        callbacks=callbacks
        )



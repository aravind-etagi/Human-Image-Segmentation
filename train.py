import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from glob import glob
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle


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


if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)

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






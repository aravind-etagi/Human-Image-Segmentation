import os
from glob import glob
from sklearn.model_selection import train_test_split
import numpy as np

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path, test_size=0.1):
    """Takes images folder path and returns train and test data"""
    img_path  = os.path.join(path, 'images', '*.jpg')
    mask_path = os.path.join(path, 'masks', '*.png')

    img_list  = sorted(glob(img_path))
    mask_list = sorted(glob(mask_path))

    train_img, test_img = train_test_split(img_list, test_size=test_size, random_state=42)
    train_mask, test_mask = train_test_split(mask_list, test_size=test_size, random_state=42)

    return (train_img, train_mask), (test_img, test_mask)


if __name__ == '__main__':
    # setting random state
    np.random.seed(42)

    # Loading dataset
    data_path = 'people_segmentation'
    (train_img, train_mask), (test_img, test_mask) = load_data(data_path)

    print(f'number of train images : {len(train_img)}, number of train masks : {len(train_mask)}')
    print(f'number of test images  : {len(test_img)} , number of test masks  : {len(test_mask)}')
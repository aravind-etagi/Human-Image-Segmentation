import os
from glob import glob
from sklearn.model_selection import train_test_split

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

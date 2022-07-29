import os
from glob import glob
from sklearn.model_selection import train_test_split
import numpy as np
import albumentations as A
import cv2
from tqdm import tqdm

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


def augument_data(images_list, masks_list, aug_dest_dir, augument=True):
    H, W = 512, 512

    for i, m in tqdm(zip(images_list, masks_list)):

        img_name = i.split(os.sep)[-1].split('.')[0]

        img  = cv2.imread(i, cv2.IMREAD_COLOR)
        mask = cv2.imread(m, cv2.IMREAD_COLOR)

        if augument:

            num_of_copies = 5

            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ToGray(p=0.25),
                A.ChannelShuffle(p=0.25),
                A.CoarseDropout(max_holes=10, min_holes=3, max_height=32, max_width=32, p=0.2),
                A.Rotate(limit=45, p=1)
                ])
            
            # aug = A.HorizontalFlip(p=1)
            # flip_transform = aug(image=img, mask=mask)
            # img_flip  = flip_transform['image']
            # mask_flip = flip_transform['mask']

            # img_grey  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # mask_grey = mask

            # aug = A.ChannelShuffle(p=1)
            # channel_shift_transform = aug(image=img, mask=mask)
            # img_chnl_shift  = channel_shift_transform['image']
            # mask_chnl_shift = channel_shift_transform['mask']

            # aug = A.CoarseDropout(max_holes=10, min_holes=3, max_height=32, max_width=32, p=1)
            # holes_transform = aug(image=img, mask=mask)
            # img_with_holes  = holes_transform['image']
            # mask_with_holes = holes_transform['mask']

            # aug = A.Rotate(limit=45, p=1)
            # rotate_transform = aug(image=img, mask=mask)
            # img_rotated  = rotate_transform['image']
            # mask_rotated = rotate_transform['mask']

            # x = [img, img_flip, img_grey, img_chnl_shift, img_with_holes, img_rotated]
            # y = [mask, mask_flip, mask_grey, mask_chnl_shift, mask_with_holes, mask_rotated]

            transformed_image_list = []
            transformed_mask_list  = []
            for index in range(5):
                transformed = transform(image=img, mask=mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['mask']

                transformed_image_list.append(transformed_image)
                transformed_mask_list.append(transformed_mask)

        else:
            transformed_image_list = [img]
            transformed_mask_list  = [mask]
        
        for index, (transformed_image, transformed_mask) in enumerate(zip(transformed_image_list, transformed_mask_list)):
            try:
                aug = A.CenterCrop(H, W, p=1.0)
                augmented = aug(image=transformed_image, mask=transformed_mask)
                i = augmented["image"]
                m = augmented["mask"]
            except Exception as e:
                i = cv2.resize(transformed_image, (W, H))
                m = cv2.resize(transformed_mask, (W, H))
                    
            cv2.imwrite(os.path.join(aug_dest_dir,'images', f'{img_name}_{index}.png'), i)
            cv2.imwrite(os.path.join(aug_dest_dir,'masks', f'{img_name}_{index}.png'), m*255)
        

if __name__ == '__main__':
    # setting random state
    np.random.seed(42)

    # Loading dataset
    data_path = 'people_segmentation'
    (train_img, train_mask), (test_img, test_mask) = load_data(data_path)

    print(f'number of train images : {len(train_img)}, number of train masks : {len(train_mask)}')
    print(f'number of test images  : {len(test_img)} , number of test masks  : {len(test_mask)}')

    create_dir('people_segmentation/augumented_data/train/images')
    create_dir('people_segmentation/augumented_data/train/masks')
    create_dir('people_segmentation/augumented_data/test/images')
    create_dir('people_segmentation/augumented_data/test/masks')

    augument_data(train_img, train_mask, 'people_segmentation/augumented_data/train')
    augument_data(test_img, test_mask, 'people_segmentation/augumented_data/test', augument=False)
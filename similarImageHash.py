import imagehash
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import cv2
import numpy as np
import time

def compute_hash(data_root, obj_list):
    for obj_name in tqdm(obj_list):

        image_paths = data_root + '/' + obj_name + '/train/good/rgb'#mvtec3d
        image_list = os.listdir(image_paths)
        aver_hash_list = []  
        per_hash_list = []  
        gradient_hash_list = [] 
        similar_image_list = []  
        for item in image_list:
            image = Image.open(image_paths + '/' + item)
            resized_image = image.resize((256, 256))

            aver_hash = imagehash.average_hash(resized_image)
            per_hash = imagehash.phash(resized_image)
            gradient_hash = imagehash.dhash(resized_image)

            aver_hash_list.append(aver_hash)
            per_hash_list.append(per_hash)
            gradient_hash_list.append(gradient_hash)

        for i, item_i in enumerate(image_list):
            min_score = float('inf')
            similar_image = ''
            for j, item_j in enumerate(image_list):
                if not i == j:
                    if abs(per_hash_list[i] - per_hash_list[j]) < min_score:
                        min_score = abs(per_hash_list[i] - per_hash_list[j])
                        similar_image = item_j
            similar_image_list.append(similar_image)
      
        data_excel = {"image_name": image_list,
                      "aver_hash": aver_hash_list,
                      "per_hash": per_hash_list,
                      "gradient_hash": gradient_hash_list,
                      "similar_image": similar_image_list
                      }

        df = pd.DataFrame(data_excel)

        df.to_excel('./hash/mvtec3d/' + obj_name + '.xlsx', index=False)



def get_similar_image(data_excel,image_path, obj_name, test_image):
 
    resize_shape = (256, 256)

    image_path_temp = os.path.abspath(image_path)
    index = image_path_temp.find(obj_name)
    image_path = image_path_temp[:index]
    test_image_pil = Image.fromarray(test_image)
    test_image_ahash = imagehash.average_hash(test_image_pil)
    test_image_phash = imagehash.phash(test_image_pil)
    test_image_dhash = imagehash.dhash(test_image_pil)

    min_score = float('inf')
    similar_image_name = None
    for i in range(0, len(data_excel["image_name"])):
        aver_hash = imagehash.hex_to_hash(data_excel["aver_hash"][i])
        per_hash = imagehash.hex_to_hash(data_excel["per_hash"][i])
        gradient_hash = imagehash.hex_to_hash(data_excel["gradient_hash"][i])
        if abs(test_image_phash - per_hash) < min_score:
            min_score = abs(test_image_phash - per_hash)
            similar_image_name = data_excel["image_name"][i]
    similar_image = cv2.imread(image_path + '/' + obj_name + '/train/good/' + similar_image_name)
    similar_image = similar_image / 255.0
    similar_image_tensor = torch.from_numpy(np.transpose(similar_image, (2, 0, 1))).float()
    return similar_image_tensor


if __name__ == "__main__":
    # obj_list = [
    #     'capsule',
    #     'bottle',
    #     'carpet',
    #     'leather',
    #     'pill',
    #     'transistor',
    #     'tile',
    #     'cable',
    #     'zipper',
    #     'toothbrush',
    #     'metal_nut',
    #     'hazelnut',
    #     'screw',
    #     'grid',
    #     'wood'
    # ]
    obj_list = [
        'bagel',
         'cable_gland',
         'carrot',
        'cookie',
        'dowel',
        'foam',
        'peach',
        'potato',
        'rope',
        'tire',
    ]
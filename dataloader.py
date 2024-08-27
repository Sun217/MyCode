import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np
from CutpasteAugment import cut_patch, paste_patch
from torchvision import transforms
import random
from PIL import Image


class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir + "/*/*.png")) # mvtec
        # self.images = sorted(glob.glob(root_dir + "/*/*.JPG")) # visa
        # self.images = sorted(glob.glob(root_dir + "/*" + "/rgb" + "/*.png"))# mvtec3d
        self.resize_shape = resize_shape

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)
        imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        imagegray = imagegray[:, :, None]
        image = np.transpose(image, (2, 0, 1))  # z,x,y
        imagegray = np.transpose(imagegray, (2, 0, 1))

        mask = np.transpose(mask, (2, 0, 1))
        return image, mask, imagegray

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        parent_dir = os.path.dirname(dir_path)
        if base_dir == 'good':
        # if parent_dir[-4:] == 'good':  # mvtec3d
            image, mask, imagegray = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            # mask_path = os.path.join(parent_dir, 'gt/')  # mvtec-3d
            mask_path = os.path.join(dir_path, '../../ground_truth/')#mvtec
            mask_path = os.path.join(mask_path, base_dir)#mvtec
            file_name = file_name.split(".")[0]
            # file_name = file_name[-3:]
            # mask_file_name = file_name + ".png"  # mvtec-3d,visa
            mask_file_name = file_name + "_mask.png"#mvtec
            # mask_file_name = file_name.split(".")[0]+".png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask, imagegray = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly, 'mask': mask, 'idx': idx, 'imagegray': imagegray,
                  'image_path': img_path}
        return sample


class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self, root_dir, anomaly_source_path, resize_shape=None, stage='one'):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.stage = stage
        self.root_dir = root_dir
        self.resize_shape = resize_shape

        self.image_paths = sorted(glob.glob(root_dir + "/*.png"))# mvtec
        # self.image_paths = sorted(glob.glob(root_dir + "/*.JPG")) #visa
        # self.image_paths = sorted(glob.glob(root_dir + "/*.bmp")) #btad
        self.anomaly_source_paths = self.image_paths

        # self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                                              children=iaa.WithChannels(0, iaa.Add((0, 90)))),
                           iaa.Dropout2d(p=0.5),  
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),  
                           iaa.pillike.EnhanceSharpness(),  
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),  
                           iaa.Solarize(0.5, threshold=(32, 128)),  
                           iaa.Posterize(),  
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]
        colorJitter = 0.5
        self.colorJitter = transforms.ColorJitter(brightness=colorJitter,
                                                  contrast=colorJitter,
                                                  saturation=colorJitter,
                                                  hue=colorJitter)
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def __len__(self):
        return len(self.image_paths)

    def Diffuse(self, img):
        h = img.shape[0]
        w = img.shape[1]

        mask = np.zeros((h, w), dtype=np.uint8)
        img = img * 255
        augmented = Image.fromarray(np.uint8(img))
        img = Image.fromarray(img.astype('uint8'))

        for i in range(2):
            cut_w, cut_h = int(h * 0.05), int(h * 0.05)  # 生成异常区域数
            fx = int(random.uniform(0, 4))
            num = 0
            bj = 1
            c = int(random.uniform(0, h))
            k = int(random.uniform(0, w))
            while num < (4):  
                insert_box = [k, c, k + cut_w, c + cut_h]
                from_location_h = int(random.uniform(0, h - cut_h))
                from_location_w = int(random.uniform(0, w - cut_w))
                box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
                rnd = random.randint(0, 4)
                fw = 2
                no_anomaly = torch.rand(1).numpy()[0]
                if no_anomaly > 0.7:
                    box = insert_box
                if rnd == 4:
                    box = [from_location_w, from_location_h, from_location_w + cut_w * fw, from_location_h + cut_h * fw]
                    insert_box = [k, c, k + cut_w * fw, c + cut_h * fw]
                    rnd = random.randint(0, 3)
                    if no_anomaly > 0.7:
                        box = insert_box
                if rnd == 0:
                    if no_anomaly > 0.7:
                        box = insert_box
                        patch = img.crop(box)
                        patch = self.colorJitter(patch)
                    else:
                        box = insert_box
                        patch = img.crop(box)
                elif rnd == 1 or rnd == 3:
                    patch = img.crop(box)
                    bright_arr = np.array(patch)
                    bright_arr[:, :, 0] = 100
                    bright_arr = bright_arr.astype('uint8')
                    patch = Image.fromarray(bright_arr)
                    patch = self.colorJitter(patch)
                elif rnd == 2:
                    patch = img.crop(box)
                    patch = self.colorJitter(patch)
                augmented.paste(patch, insert_box)
                mask[insert_box[1]:insert_box[3], insert_box[0]:insert_box[2]] = 255
                fx1 = fx
                fx = int(random.uniform(0, 4))
                while fx1 - fx == abs(2):
                    fx = int(random.uniform(0, 4))
                if fx == 0:
                    c = c - cut_w
                elif fx == 2:
                    c = c + cut_w
                elif fx == 1:
                    k = k - cut_h
                elif fx == 3:
                    k = k + cut_h
                num += 1
                if (c > 0 and c < h and k > 0 and k < w) == False:
                    c = int(random.uniform(0, h))
                    k = int(random.uniform(0, w))
        mask = np.expand_dims(mask, axis=2)
        mask = (mask / 255.0).astype(np.float32)
        has_anomaly = 1.0
        if np.sum(mask) == 0:
            has_anomaly = 0.0
        augmented = np.array(augmented)
        augmented = augmented.astype(np.float32) / 255.0
        return augmented, mask, np.array([has_anomaly], dtype=np.float32)

    def randAugmenter(self):  
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        """

        Args:
            image:
            anomaly_source_path: 主函数传入的异常数据源（与训练数据同源）
        Returns:

        """
        aug = self.randAugmenter()
        # [2^min_perlin_scale,2^perlin_scale ]
        # perlin_scale = 6 # 小尺度
        # min_perlin_scale = 3
        perlin_scale = 3 # 大尺度
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        
        block_num = 2
        image_h, image_w = anomaly_source_img.shape[0], anomaly_source_img.shape[1]
        block_size = (int(image_h / block_num), int(image_w / block_num))
        step_size = block_size

        block_list = []
        for y in range(0, anomaly_source_img.shape[0], step_size[1]):
            for x in range(0, anomaly_source_img.shape[1], step_size[0]):
                block = anomaly_source_img[y:y + block_size[1], x:x + block_size[0]]
                block_list.append(block)

        index_matrix = np.random.permutation(block_num * block_num).reshape((block_num, block_num))
        image_col = []
        for row in index_matrix:
            image_row = []
            for j in row:
                image_row.append(block_list[j])
            image_col.append(cv2.hconcat(image_row))


        anomaly_source_img = cv2.vconcat(image_col)
        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        threshold = 0.5
        while True:
            perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
            perlin_noise = self.rot(image=perlin_noise)
            perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            if perlin_thr.sum() > 50:
                break
        perlin_thr = np.expand_dims(perlin_thr, axis=2)
        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0
        beta = torch.rand(1).numpy()[0] * 0.8
        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (perlin_thr)
        no_anomaly = torch.rand(1).numpy()[0]

        if no_anomaly > 2:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0], dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1 - msk) * image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly = 0.0
            return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)
    
    def augment_and_diffuse(self, image, anomaly_source_path):
        # augmented_image, augmented_mask, has_anomaly = self.Diffuse(image)
        augmented_image, augmented_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        if self.stage == 'one':
            return augmented_image, augmented_mask, has_anomaly
        elif self.stage == 'two':
            aug_diff_image, diff_mask, diff_has_anomaly = self.Diffuse(augmented_image)
            merged_mask = np.add(augmented_mask, diff_mask)  
            merged_mask = np.clip(merged_mask, 0.0, 1.0)  
            # return augmented_image, augmented_mask, aug_diff_image, merged_mask, has_anomaly
            return aug_diff_image, merged_mask, diff_has_anomaly


    def augment_cutpaste(self, image):
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 2:  
            image = image.astype(np.float32)
            return image, np.zeros((image.shape[0], image.shape[1], 1), dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:  # 执行cutpaste

            patch = cut_patch(image)
            augmented_image, msk = paste_patch(image, patch)
            msk = msk.astype(np.float32)
            augmented_image = augmented_image.astype(np.float32)
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly = 0.0
            return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        try:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        except:
            print(str(image_path))

        import random
        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        what_anomaly = torch.rand(1).numpy()[0]
        kai = False
        if what_anomaly > 2:
            augmented_image, anomaly_mask, has_anomaly = self.augment_cutpaste(image)  # Cutpaset
            if kai :
                augm1 = augmented_image * 255
                augm2 = anomaly_mask * 255
                r = ''.join(str(random.randint(0, 100)) for _ in range(5))
                filename = r + "ima" + ".png"
                filenamemas = r + "mas" + ".png"
                cv2.imwrite("./augment/" + filename, augm1)
                cv2.imwrite("./augment/" + filenamemas, augm2)

        else:
            augmented_image, anomaly_mask, has_anomaly = self.augment_and_diffuse(image, image_path)
            if kai == 1:
                augm1 = augmented_image * 255
                augm2 = anomaly_mask * 255
                r = ''.join(str(random.randint(0, 100)) for _ in range(5))
                filename = r + "ima" + ".png"
                filenamemas = r + "mas" + ".png"
                cv2.imwrite("./augment/" + filename, augm1)
                cv2.imwrite("./augment/" + filenamemas, augm2)


        auggray = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2GRAY)
        auggray = auggray[:, :, None]
        image = np.transpose(image, (2, 0, 1))
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        auggray = np.transpose(auggray, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))

        return image, augmented_image, anomaly_mask, has_anomaly, auggray

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        image, augmented_image, anomaly_mask, has_anomaly, auggray = self.transform_image(self.image_paths[idx],
                                                                                          self.anomaly_source_paths[
                                                                                              anomaly_source_idx])

        sample = {'image': image, "anomaly_mask": anomaly_mask, 'augmented_image': augmented_image,
                  'has_anomaly': has_anomaly, 'idx': idx, 'auggray': auggray}

        return sample

import numpy as np
from scipy.signal import find_peaks
import cv2
from utilts_custom_class import *
from scipy.ndimage import median_filter as med_filt
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed

import torch
import torch.nn.functional as F
from numpy import ndarray as NDArray
from skimage import measure
from sklearn.metrics import auc, roc_auc_score, roc_curve
from tqdm import tqdm
from statistics import mean
import random
def residual_th(res_img, threshold_per = 0.05):
    try:
        rows,cols,_  	= res_img.shape
        residual_abs  	= np.array(np.abs(res_img)*255,dtype=np.uint8).reshape((rows,cols,3))
    except:
        rows,cols     	= res_img.shape
        residual_abs  	= np.array(np.abs(res_img)*255,dtype=np.uint8).reshape((rows,cols,1))
    
    
    unique, counts 	= np.unique(residual_abs, return_counts=True)
    count_th_1      = counts<=(threshold_per*np.max(counts))
    counts_th       = dict(zip(unique, count_th_1))
    image_res_th 	= np.array(residual_abs)
    
    intensity_n_zer = np.array(list(counts_th.values()))*np.array(list(counts_th.keys()))
    intensity_n_zer = intensity_n_zer[np.nonzero(intensity_n_zer)]
    val_th          = np.isin(image_res_th,intensity_n_zer)
    img_th          = residual_abs* val_th
    
    '''
    for i in range(rows):
        for j in range(cols):

            if len(residual_abs.shape)==3:
                value       = residual_abs[i,j,0]
            else:
                value       = residual_abs[i,j]

            counts_int  = counts_th[value]
            if counts_int==0:                             #if gray_img_np[i,j]>=60 and gray_img_np[i,j]<=215:
                image_res_th[i,j]   = 0                           #gray_img_np[i,j] = 255
            else:
                pass
    return image_res_th'''

    return img_th

def seg_module(orig_batch, res_batch, th_pix=0.95):
    """
    这段代码是一个图像分割处理模块，用于对输入图片进行分割并对分割后的区域进行裁剪。具体实现过程如下：
    """
    #sig = nn.Sigmoid()
    # 将原始图像和残差图像转换为 numpy 数组；
    batch_new = res_batch.detach().cpu().numpy()#掩码

    batch_org = orig_batch.detach().cpu().numpy()#异常图片
    # 对每张图片
    for i in range(res_batch.shape[0]):
        # 进行阈值化处理（根据像素强度），得到二值化的残差图像；
        image_residual_n        =       batch_new[i].reshape(256,256)
        image_residual_th       =       residual_th(image_residual_n,threshold_per=th_pix)
        # 对残差图像进行中值滤波；
        batch_org[i]              =       med_filt(image_residual_th,size=3)
    return batch_org

def crop_area(orig_img, mask):
    #orig_img        =           orig_img.reshape((256,256,3))
    # 这行代码实现了根据掩码来选择并保留原始图像中特定区域的像素值，而将其他区域的像素值置为零。
    for i in range(3):
        orig_img[i,:,:]         =        orig_img[i,:,:]*(mask>0)

    return orig_img

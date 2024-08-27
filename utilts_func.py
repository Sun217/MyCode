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
    


    return img_th

def seg_module(orig_batch, res_batch, th_pix=0.95):
    batch_new = res_batch.detach().cpu().numpy()
    batch_org = orig_batch.detach().cpu().numpy()
    for i in range(res_batch.shape[0]):
        image_residual_n        =       batch_new[i].reshape(256,256)
        image_residual_th       =       residual_th(image_residual_n,threshold_per=th_pix)
        batch_org[i]              =       med_filt(image_residual_th,size=3)
    return batch_org

def crop_area(orig_img, mask):
    for i in range(3):
        orig_img[i,:,:]         =        orig_img[i,:,:]*(mask>0)

    return orig_img

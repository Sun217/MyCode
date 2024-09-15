import time
from concurrent.futures import ThreadPoolExecutor

import torch
from dataloader import MVTecDRAEMTrainDataset, MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
from torch import optim
# from tensorboard_visualizer import TensorboardVisualizer
from model_unetskip import ReconstructiveSubNetwork
from loss import SSIM
from segmodel_two import Seg_Network2
import os
import kornia
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unetskip import ReconstructiveSubNetwork
import os
import cv2
from utils import savefig
from msgms import MSGMSLoss
import kornia
from ColorlossLab import ColorDifference
import torchvision.transforms.functional as TF
from torchvision import transforms
import copy
from seg_model import Seg_Network
import pandas as pd
import imagehash
from PIL import Image
from torchvision.transforms import ToPILImage

toPIL = transforms.ToPILImage()
from pro_curve_util import compute_pro
from generic_util import trapezoid
from test_new import load_and_process_train_image,get_similar_image_hash

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def mse(image1, image2):
    return np.mean((image1 - image2) ** 2)


def mean_smoothing(amaps, kernel_size: int = 21):
    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
    mean_kernel = mean_kernel.to(amaps.device)
    return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)


def one_mask_process(pre_mask_batch, Bin_threshold=0.8, min_area=7000):

    pre_mask_batch = (pre_mask_batch > Bin_threshold).float()  
    One_mask_batch = pre_mask_batch.detach().cpu().numpy()  
    image_residual_th = One_mask_batch[0].reshape((256, 256)) * 255
    image_residual_th = image_residual_th.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_residual_th, connectivity=8)
    for label in range(1, num_labels):  
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:  
            image_residual_th[labels == label] = 0
    One_mask_batch[0] = image_residual_th / 255.0

    pre_mask_batch = torch.from_numpy(One_mask_batch).cuda()
    return pre_mask_batch


def train_on_device(obj_names, args):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    for obj_name in obj_names:
        data_excel = pd.read_excel('./hash/' + obj_name + '.xlsx')
        similar_image_names = data_excel["image_name"].tolist()

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(load_and_process_train_image, "E:/mvtec_anomaly_detection/", obj_name, img_name,
                                resize_shape=(256, 256))
                for img_name in similar_image_names
            ]
            train_image_tensors_list = [future.result() for future in futures]
        run_name = 'mvtec_' + str(args.lr) + '_' + str(args.epochs) + '_bs' + str(args.bs) + "_" + obj_name + '_'
        model_one_name = 'EdgRec_' + str(args.lr) + '_' + '600' + '_bs4' + "_" + obj_name + '_'

        model_one = Seg_Network(in_channels=3, out_channels=1)
        model_one = model_one.cuda()
        model_one.load_state_dict(torch.load('.1/bs4/' + model_one_name + 'best_1' + '.pckl',map_location='cpu'))  # map_location参数用于指定将权重加载到哪个设备上。
       
        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.cuda()
        model.apply(weights_init)

        optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.lr}, ], weight_decay=0)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args.epochs * 0.4, args.epochs * 0.7], gamma=0.2,
                                                   last_epoch=-1)
        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM()
        img_dim = 256
        dataset = MVTecDRAEMTrainDataset(args.data_path + obj_name + "/train/good/", args.anomaly_source_path,resize_shape=[256, 256], stage='two')
        dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=0)

        dataset_test = MVTecDRAEMTestDataset(args.data_path + obj_name + "/test/", resize_shape=[img_dim, img_dim])
        dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
        kernel = torch.ones(3, 3).cuda()
        aucm = 0
        aucm_pixel = 0
        PRO_max = 0
        max_ap_pixel = 0
        lun = 0

        Resume = False
        if Resume:
            path = 'E:/sjk/two_stage/.2/EdgRec_0.0001_600_bs4_screw_last_2.pckl'
            checkpoint = torch.load(path, map_location=('cpu'))
            model.load_state_dict(checkpoint)
        if obj_name != '03':
            epochs = args.epochs
        else:
            epochs = 100
        for epoch in range(epochs):
            mask_cnt = 0
            Loss = 0
            model.train()
            j = 0
            for i_batch, sample_batched in enumerate(dataloader):
                ori_image = sample_batched["image"].cuda()  
                anomaly_mask = sample_batched["anomaly_mask"].cuda()  
                augmented_image = sample_batched["augmented_image"].cuda()  
                input_image = torch.empty_like(ori_image)

                pre_mask_batch = model_one(augmented_image)  
                pre_mask_batch = one_mask_process(pre_mask_batch, 0.8, 7000)

                good_image = ori_image[0]
                pre_mask = pre_mask_batch[0]
                input_image[0] = torch.where(pre_mask > 0, good_image, augmented_image[0])
                recon_image = model(input_image)

                l2_loss = loss_l2(recon_image, ori_image)
                ssim_loss = loss_ssim(recon_image, ori_image)
                loss = l2_loss + ssim_loss
                Loss = loss + Loss  
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('obj_name:{:}  epoch:{:}  Loss:{:.4f}'.format(obj_name, epoch, Loss.data))
            scheduler.step()

            if epoch % 2 == 0:
                model.eval()
                pro_gt = []
                pro_out = []
                anomaly_score_gt = []
                anomaly_score_prediction = []
                total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset_test)))
                total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset_test)))
                msgms = MSGMSLoss().cuda()
                kernel = torch.ones(3, 3).cuda()
                with torch.no_grad():
                    i = 0
                    for i_batch, sample_batched in enumerate(dataloader_test):
                        ori_image = sample_batched["image"].cuda()
                        is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]
                        anomaly_score_gt.append(is_normal)
                        true_mask = sample_batched["mask"]
                        true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

                        input_image = torch.empty_like(ori_image)

                        pre_mask_batch = model_one(ori_image)
                        pre_mask_batch = one_mask_process(pre_mask_batch, 0.8, 7000)
                        similar_good_image = get_similar_image_hash(data_excel, train_image_tensors_list,image_test=ori_image[0]) 
                        similar_good_image = similar_good_image.cuda()

                        pre_mask = pre_mask_batch[0]
                        input_image[0] = torch.where(pre_mask > 0, similar_good_image, ori_image[0])

                        gray_rec = model(input_image)
                        recimg = (gray_rec.detach().cpu().numpy()[0].transpose((1, 2, 0)) * 180).astype('uint8')
                        oriimg = (ori_image.detach().cpu().numpy()[0].transpose((1, 2, 0)) * 180).astype('uint8')
                        colorD = ColorDifference(recimg, oriimg)
                        mgsgmsmap = msgms(gray_rec, ori_image, as_loss=False)
                        mgsgmsmapmean = mean_smoothing(mgsgmsmap, 21)
                        out_mask_gradient = mgsgmsmapmean.detach().cpu().numpy()
                        out_mask_averaged = colorD[None, None, :, :] + out_mask_gradient
                        image_score = np.max(out_mask_averaged)
                        anomaly_score_prediction.append(image_score)
                        # ==========================================
                        flat_true_mask = true_mask_cv.flatten()
                        flat_out_mask = out_mask_averaged.flatten()
                        total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
                        total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
                        mask_cnt += 1
                        # ===========PRO=========================
                        # truegt = true_mask_cv[:, :, 0]
                        # outresult = out_mask_averaged[0, 0, :, :]
                        # pro_gt.append(truegt)
                        # pro_out.append(outresult)
                # ==========PRO=========
                # all_fprs, all_pros = compute_pro(anomaly_maps=pro_out, ground_truth_maps=pro_gt, num_thresholds=5000)
                # au_pro = trapezoid(all_fprs, all_pros, x_max=0.3)
                # au_pro /= 0.3

                # anomaly_score_prediction = np.array(anomaly_score_prediction)
                # anomaly_score_gt = np.array(anomaly_score_gt)
                # auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
                total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
                total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
                total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
                # auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
                ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
                if ap_pixel > max_ap_pixel:
                    max_ap_pixel = ap_pixel
                    lun=epoch
                    # torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name + "best_2" + ".pckl"))
                # torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name + "last_2" + ".pckl"))
                # # if auroc + auroc_pixel > aucm + aucm_pixel:
                # if PRO_max<au_pro:
                #     PRO_max=au_pro
                #     torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name + "best_2" + ".pckl"))
                # if aucm_pixel < auroc_pixel:
                #     aucm_pixel = auroc_pixel
                # if auroc > aucm or (auroc == aucm and auroc_pixel >= aucm_pixel):
                #     # torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name + "best_2" + ".pckl"))
                #     aucm = auroc
                #     # aucm_pixel = auroc_pixel
                #     lun = epoch
                print("Current epoch AP_pixel: " + str(ap_pixel) + "Last saved epoch: " + str(lun) + "max_ap_pixel: " + str(max_ap_pixel))
                # print("Current epoch Auc_image/Auc_pixel/PRO:" + str(auroc) + "   " + str(auroc_pixel) +str(au_pro) + "\n"
                #       + "Last saved epoch:" + str(lun) + " " + "the Auc_image/Auc_pixel:" + str(aucm) + "  " + str(
                #     aucm_pixel)+" max pixel_auroc:"+str(aucm_pixel)+ "max PRO: "+str(PRO_max))
                # if auroc == 1 and epoch - lun > 200:
                #     break


if __name__ == "__main__":
    if __name__ == "__main__":
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--obj_id', action='store', type=int, default='-1', required=False)
        parser.add_argument('--bs', action='store', type=int, default='4', required=False)
        parser.add_argument('--lr', action='store', type=float, default='0.0001', required=False)
        parser.add_argument('--epochs', action='store', type=int, default='300', required=False)
        parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
        # parser.add_argument('--data_path', action='store', type=str, default='E:/mvtec3d_mvtec/',required=False)
        parser.add_argument('--data_path', action='store', type=str, default='E:/mvtec_anomaly_detection/',
                            required=False)
        # parser.add_argument('--anomaly_source_path', action='store', type=str, default='E:/mvtec3d_mvtec/', required=False)
        parser.add_argument('--anomaly_source_path', action='store', type=str, default='E:/mvtec_anomaly_detection/',
                            required=False)
        parser.add_argument('--checkpoint_path', action='store', type=str, default='.2/bs4/AP/', required=False)
        parser.add_argument('--log_path', action='store', type=str, default='./Mod', required=False)
        parser.add_argument('--visualize', action='store_true', default=True)
        args = parser.parse_args()
        obj_batch = [['capsule'],
                     ['bottle'],
                     ['carpet'],
                     ['leather'],
                     ['pill'],
                     ['transistor'],
                     ['tile'],
                     ['cable'],
                     ['zipper'],
                     ['toothbrush'],
                     ['metal_nut'],
                     ['hazelnut'],
                     ['screw'],
                     ['grid'],
                     ['wood']
                     ]

        if int(args.obj_id) == -1:
            obj_list = [
                # 'carpet',
                # 'leather',
                # 'pill',
                # 'transistor',
                # 'capsule',
                # 'screw',
                # 'tile',
                'cable',
                # 'zipper',
                'toothbrush',
                'metal_nut',
                'hazelnut',
                'grid',
                'wood',
                'bottle',
            ]
            # obj_list = [
            #     # 'bagel',
            #     #  'cable_gland',
            #     #  # 'carrot',
            #     #  'cookie',
            #     #  'dowel',
            #     # # 'foam',
            #     # 'peach',
            #     # # 'potato',
            #     # 'rope',
            #     'tire',
            # ]
            # obj_list = [# visa
            #     'candle',
            #     'capsules',
            #     'cashew',
            #     'chewinggum',
            #     'fryum',
            #     'macaroni1',
            #     'macaroni2',
            #     'pcb1',
            #     'pcb2',
            #     'pcb3',
            #     'pcb4',
            #     'pipe_fryum',
            # ]
            # obj_list = [  # btad
            #     # '01',
            #     # '03',
            #     # '02',
            # ]
            picked_classes = obj_list
        else:
            picked_classes = obj_batch[int(args.obj_id)]

        with torch.cuda.device(args.gpu_id):
            train_on_device(picked_classes, args)

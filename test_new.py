import torch
import torch.nn.functional as F
from dataloader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unetskip import ReconstructiveSubNetwork
import os
import cv2
from utils import savefig
from msgms import MSGMSLoss
import kornia
import pandas as pd
from seg_model import Seg_Network
from ColorlossLab import ColorDifference
from generic_util import trapezoid
from pro_curve_util import compute_pro
from loss import SSIM
from torchvision import transforms
from utilts_func import residual_th
from datetime import datetime
from scipy.ndimage import median_filter as med_filt
from MultiResUNet import MultiResUnet
import random
from torchvision.transforms import ToPILImage
import imagehash
import time
from concurrent.futures import ThreadPoolExecutor
toPIL = transforms.ToPILImage()
from torchsummary import summary


def see_img(data, dir, i, type):  
    data = data.permute(0, 2, 3, 1)
    data = data[0, :, :, :]
    data = data.cpu().numpy()
    data = data * 200
    data = data.astype('uint8')
    cv2.imwrite(dir + '/' + f'{type}{i}.png', data)


def save_tensor_image(tensor, save_path):
    temp_tensor = tensor[0]
    np_arr = temp_tensor.cpu().numpy()
    np_arr = np_arr * 220
    np_arr = np.transpose(np_arr, (1, 2, 0))
    cv2.imwrite(save_path, np_arr)


def see_img_heatmap(data, segresult, dir, i, type):  
    y2max = 255
    y2min = 0
    x2max = segresult.max()
    x2min = segresult.min()
    segresult = np.round((y2max - y2min) * (segresult - x2min) / (x2max - x2min) + y2min)
    segresult = segresult.astype(np.uint8)
    heatmap = cv2.applyColorMap(segresult, colormap=cv2.COLORMAP_JET)
    alpha = 0.15
    alpha2 = 0.3
    data = data.permute(0, 2, 3, 1)
    data = data[0, :, :, :]
    data = data.cpu().numpy()
    data = data * 200
    data = data.astype('uint8')
    overlay = data.copy()
    data = cv2.addWeighted(heatmap, alpha2, overlay, 1 - alpha, 0, overlay)
    cv2.imwrite(dir + '/' + f'{type}{i}.png', data)


def write_results_to_file(run_name, image_auc, pixel_auc, image_ap, pixel_ap):
    if not os.path.exists('./outputs/'):
        os.makedirs('./outputs/')

    # 获取当前时间
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    fin_str = formatted_time + '\n'
    fin_str = fin_str + "img_auc," + run_name
    for i in image_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += "," + str(np.round(np.mean(image_auc), 3))
    fin_str += "\n"
    fin_str += "pixel_auc," + run_name
    for i in pixel_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += "," + str(np.round(np.mean(pixel_auc), 3))
    fin_str += "\n"
    fin_str += "img_ap," + run_name
    for i in image_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += "," + str(np.round(np.mean(image_ap), 3))
    fin_str += "\n"
    fin_str += "pixel_ap," + run_name
    for i in pixel_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += "," + str(np.round(np.mean(pixel_ap), 3))
    fin_str += "\n"
    fin_str += "--------------------------\n"

    with open("./outputs/results.txt", 'a+') as file:
        file.write(fin_str)



def load_and_process_train_image(image_path, obj_name, similar_image_name, resize_shape=(256,256)):
    img = cv2.imread(f"{image_path}/{obj_name}/train/good/{similar_image_name}")
    img = cv2.resize(img, resize_shape) / 255.0
    return torch.from_numpy(np.transpose(img, (2, 0, 1))).float()

def get_similar_image_hash(data_excel, train_image_tensors_list, image_test):

    test_image_pil = ToPILImage()(image_test)
    test_image_phash = imagehash.phash(test_image_pil)
    per_hashes = np.array([imagehash.hex_to_hash(h) for h in data_excel["per_hash"]])
    differences = np.abs(per_hashes - test_image_phash)
    min_score_index = np.argmin(differences)
    similar_image_tensor = train_image_tensors_list[min_score_index]

    return similar_image_tensor


def mean_smoothing(amaps, kernel_size: int = 21):
    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
    mean_kernel = mean_kernel.to(amaps.device)
    return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)


def test(obj_names, mvtec_path, checkpoint_path, base_model_name, saveimages):
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_pro_list = []
    obj_auroc_image_list = []
    for obj_name in obj_names:
        img_dim = 256
        run_name = base_model_name + "_" + obj_name + '_'
        data_excel = pd.read_excel('./hash/' + obj_name + '.xlsx')
        similar_image_names=data_excel["image_name"].tolist()
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(load_and_process_train_image, "E:/mvtec_anomaly_detection/", obj_name, img_name, resize_shape=(256,256))
                for img_name in similar_image_names
            ]
            train_image_tensors_list = [future.result() for future in futures]

        model_one = Seg_Network(in_channels=3, out_channels=1)
        model_one = model_one.cuda().eval()
        model_one.load_state_dict(
            torch.load('./.1/bs4/' + 'EdgRec_0.0001_600_bs4_' + obj_name + '_best_1' + '.pckl', map_location='cuda:0'))

        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.load_state_dict(
            torch.load('./.2/bs4/' + 'EdgRec_0.0001_600_bs4_' + obj_name + '_best_2' + '.pckl', map_location='cuda:0'))
        model.cuda()
        model.eval()
        dataset = MVTecDRAEMTestDataset(mvtec_path + obj_name + "/test/", resize_shape=[img_dim, img_dim])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

        total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        mask_cnt = 0

        summary(model=model_one, input_size=(3, 256, 256), device='cuda')
        summary(model=model, input_size=(1, 256, 256), device='cuda')

        # calculate pro
        pro_gt = []
        pro_out = []
        anomaly_score_gt = []
        anomaly_score_prediction = []

        msgms = MSGMSLoss().cuda()
        kernel = torch.ones(3, 3).cuda()
        with torch.no_grad():
            i = 0
            if not os.path.exists(f'{savepath}/{obj_name}'):
                os.makedirs(f'{savepath}/{obj_name}')

            count_time = 0
            sum_time=0

            for i_batch, sample_batched in enumerate(dataloader):
                count_time += 1
                ori_image = sample_batched["image"].cuda()
                is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]
                anomaly_score_gt.append(is_normal)
                true_mask = sample_batched["mask"]
                true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))
                input_image = torch.empty_like(ori_image)
                pre_mask_batch = model_one(ori_image)

                pre_mask_batch = (pre_mask_batch > 0.8).float().cuda()
                One_mask_batch = pre_mask_batch.detach().cpu().numpy()  # 掩码
                image_residual_th = (One_mask_batch[0].reshape((256, 256)) * 255).astype(np.uint8)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_residual_th,connectivity=8)
                for label in range(1, num_labels):  
                    area = stats[label, cv2.CC_STAT_AREA]
                    if area < 7000:  
                        image_residual_th[labels == label] = 0
                One_mask_batch[0] = image_residual_th / 255.0
                One_mask_batch = torch.from_numpy(One_mask_batch).cuda()

                similar_good_image = get_similar_image_hash(data_excel, train_image_tensors_list,image_test=ori_image[0])  # mvtec 3d
                similar_good_image = similar_good_image.cuda()
                pre_mask = One_mask_batch[0]

                input_image[0] = torch.where(pre_mask > 0, similar_good_image, ori_image[0])
                rec = model(input_image)

                recimg = (rec.detach().cpu().numpy()[0].transpose((1, 2, 0)) * 180).astype('uint8')
                oriimg = (ori_image.detach().cpu().numpy()[0].transpose((1, 2, 0)) * 180).astype('uint8')

                colorD = ColorDifference(recimg, oriimg)  # color
                mgsgmsmap = msgms(rec, ori_image, as_loss=False)
                mgsgmsmapmean = mean_smoothing(mgsgmsmap, 21)
                out_mask_gradient = mgsgmsmapmean.cpu().numpy()
                out_mask_averaged = colorD[None, None, :, :] + out_mask_gradient
                image_score = np.max(out_mask_averaged)
                anomaly_score_prediction.append(image_score)
                flat_true_mask = true_mask_cv.flatten()
                flat_out_mask = out_mask_averaged.flatten()
                total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
                total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
                mask_cnt += 1

                # for pro
                truegt = true_mask_cv[:, :, 0]
                outresult = out_mask_averaged[0, 0, :, :]
                pro_gt.append(truegt)
                pro_out.append(outresult)

        all_fprs, all_pros = compute_pro(anomaly_maps=pro_out, ground_truth_maps=pro_gt, num_thresholds=5000)
        au_pro = trapezoid(all_fprs, all_pros, x_max=0.3)
        au_pro /= 0.3
        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)
        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
        ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

        total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
        total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
        obj_ap_pixel_list.append(ap_pixel)
        obj_auroc_pixel_list.append(auroc_pixel)
        obj_auroc_image_list.append(auroc)
        obj_ap_image_list.append(ap)
        obj_pro_list.append(au_pro)
        print(obj_name)
        print("AUC Image:  " + str(auroc))
        print("AP Image:  " + str(ap))
        print("AUC Pixel:  " + str(auroc_pixel))
        print("AP Pixel:  " + str(ap_pixel))
        print("PRO:  " + str(au_pro))
        print("==============================")

    print(run_name)
    print("AUC Image mean:  " + str(np.mean(obj_auroc_image_list)))
    print("AP Image mean:  " + str(np.mean(obj_ap_image_list)))
    print("AUC Pixel mean:  " + str(np.mean(obj_auroc_pixel_list)))
    print("AP Pixel mean:  " + str(np.mean(obj_ap_pixel_list)))
    print("PRO mean:  " + str(np.mean(obj_pro_list)))
    write_results_to_file(run_name, obj_auroc_image_list, obj_auroc_pixel_list, obj_ap_image_list, obj_ap_pixel_list)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--base_model_name', action='store', type=str, default='mvtec_0.001_600_bs4', required=False)
    # parser.add_argument('--data_path', action='store', type=str, default='E:\\btad_mvtec/', required=False)
    parser.add_argument('--data_path', action='store', type=str, default='E:/mvtec_anomaly_detection/', required=False)
    parser.add_argument('--checkpoint_path', action='store', type=str, default='E:/sjk/two_stage/Mod/mvtec/', required=False)
    parser.add_argument('--saveimages', default='True', action='store_true', )
    args = parser.parse_args()
    savepath = args.checkpoint_path

    obj_list = [
        'capsule',
        # 'bottle',
        # 'carpet',
        # 'leather',
        # 'pill',
        # 'transistor',
        # 'tile',
        # 'cable',
        # 'zipper',
        # 'toothbrush',
        # 'metal_nut',
        # 'hazelnut',
        # 'screw',
        # 'grid',
        # 'wood'
    ]
    # obj_list = [
    #     # 'bagel',
    #      'cable_gland',
    #      # 'carrot',
    #      'cookie',
    #      'dowel',
    #     # 'foam',
    #     'peach',
    #     # 'potato',
    #     'rope',
    #     'tire',
    # ]
    # obj_list = [  # btad
    #     '01',
    #     '03',
    #     '02',
    # ]
    with torch.cuda.device(args.gpu_id):
        test(obj_list, args.data_path, args.checkpoint_path, args.base_model_name, args.saveimages)

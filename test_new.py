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
toPIL = transforms.ToPILImage()


def see_img(data, dir, i, type):  # 用来查看被转化为-1到1之间Tensor的图像B,C,H,W
    data = data.permute(0, 2, 3, 1)
    data = data[0, :, :, :]
    data = data.cpu().numpy()
    data = data * 200
    data = data.astype('uint8')
    cv2.imwrite(dir + '/' + f'{type}{i}.png', data)


def save_tensor_image(tensor,save_path):
    temp_tensor = tensor[0]
    np_arr = temp_tensor.cpu().numpy()
    np_arr=np_arr*220
    np_arr = np.transpose(np_arr, (1, 2, 0))
    cv2.imwrite(save_path,np_arr)


def see_img_heatmap(data, segresult, dir, i, type):  # 用来查看被转化为-1到1之间Tensor的图像B,C,H,W
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


def get_similar_image(image_path, obj_name):
    """
    Args:
        image_path: 'E:/mvtec_anomaly_detection//bottle/test\\broken_large'
        obj_name: bottle

    Returns:相似图片的tensor格式

    """
    resize_shape = (256, 256)
    # 1.处理路径
    image_path_temp = os.path.abspath(image_path)
    index = image_path_temp.find(obj_name)
    image_path = image_path_temp[index:]

    # 2.读取对应的excel 获取与相似图片的路径
    data_excle = pd.read_excel('./test_excel/test_mvtec_excel/' + obj_name + '.xlsx')
    # =================================================================
    # random_seed = random.randint(0, len(data_excle)-2)  # 消融查找相似图片
    # found_rows = data_excle.iloc[random_seed]  # 消融查找相似图片
    # similar_path = found_rows.values[1]
    # =================================================================
    found_rows = data_excle[data_excle['original_image'] == image_path]
    similar_path = found_rows['similar_image'].values[0]
    similar_path = image_path_temp[:index] + similar_path

    # 3.读取相似图片
    similar_image = cv2.imread(similar_path)
    similar_image = cv2.resize(similar_image, dsize=(resize_shape[1], resize_shape[0]))
    similar_image = similar_image / 255.0

    # 4.转为tensor 格式
    similar_image_tensor = torch.from_numpy(np.transpose(similar_image, (2, 0, 1))).float()
    return similar_image_tensor


def get_similar_image_hash(data_excel, image_path, obj_name, image_test):
    """
    Args:
        image_path: 'E:\\Mvtec3d-ad\\'
        obj_name: bottle
    Returns:相似图片的tensor格式
    """
    resize_shape = (256, 256)
    # 1.处理路径
    image_path_temp = os.path.abspath(image_path)
    index = image_path_temp.find(obj_name)
    image_path = image_path_temp[:index]
    # image_list = os.listdir(image_path + '/' + obj_name + '/train/good')
    # 将输入的test_image 从np格式转为PIL格式
    test_image_pil = ToPILImage()(image_test)
    # test_image_pil = Image.open(image_path)
    # 计算 test_image的哈希值
    test_image_phash = imagehash.phash(test_image_pil)
    # 3.读取相似图片
    min_score = float('inf')
    similar_image_name = None
    for i in range(0, len(data_excel["image_name"])):
        aver_hash = imagehash.hex_to_hash(data_excel["aver_hash"][i])
        per_hash = imagehash.hex_to_hash(data_excel["per_hash"][i])
        gradient_hash = imagehash.hex_to_hash(data_excel["gradient_hash"][i])
        if abs(test_image_phash - per_hash) < min_score:
            min_score = abs(test_image_phash - per_hash)
            similar_image_name = data_excel["image_name"][i]

    # 4.转为tensor 格式
    similar_image = cv2.imread(image_path + '/' + obj_name + '/train/good/rgb/' + similar_image_name)
    similar_image = cv2.resize(similar_image, resize_shape)
    similar_image = similar_image / 255.0
    similar_image_tensor = torch.from_numpy(np.transpose(similar_image, (2, 0, 1))).float()
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
        # data_excel = pd.read_excel('./hash/' + obj_name + '.xlsx')
        # ===================================加载一阶段模型====================================
        model_one = Seg_Network(in_channels=3, out_channels=1)
        model_one = model_one.cuda().eval()
        # map_location参数用于指定将权重加载到哪个设备上。
        model_one.load_state_dict(torch.load('./.1/bs4/' + 'EdgRec_0.0001_600_bs4_' + obj_name + '_best_1' + '.pckl', map_location='cuda:0'))
        # ===================================加载二阶段模型=====================================
        model = ReconstructiveSubNetwork(in_channels=1, out_channels=3)
        model.load_state_dict(torch.load('./.2/bs4/' + 'EdgRec_0.0001_600_bs4_' + obj_name + '_best_2' + '.pckl', map_location='cuda:0'))
        model.cuda()
        model.eval()
        dataset = MVTecDRAEMTestDataset(mvtec_path + obj_name + "/test/", resize_shape=[img_dim, img_dim])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

        total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        mask_cnt = 0

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
            for i_batch, sample_batched in enumerate(dataloader):
                ori_image = sample_batched["image"].cuda()
                gray_gray = sample_batched["imagegray"].cuda()
                # gradient = kornia.morphology.gradient(gray_gray, kernel)
                is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]

                anomaly_score_gt.append(is_normal)
                true_mask = sample_batched["mask"]
                true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))
                augment_image = torch.empty_like(ori_image)
                input_image = torch.empty_like(gray_gray)

                # start_time = time.time()
                pre_mask_batch = model_one(ori_image)

                # random_numbers = [str(random.randint(0, 100)) for _ in range(5)]
                # r = ''.join(random_numbers)
                # filename = './2/' + r + "a" + ".png"
                # filenamemas = './2/' + r + "b" + ".png"
                # filenamemass = './2/' + r + "c" + ".png"
                # filenamemasss = './2/' + r + "d" + ".png"
                # filenamemassss = './2/' + r + "e" + ".png"
                # save_tensor_image(ori_image,filename)
                # save_tensor_image(pre_mask_batch,filenamemas)
                # ==================================掩码处理====================
                pre_mask_batch = (pre_mask_batch > 0.7).float()
                One_mask_batch = pre_mask_batch.detach().cpu().numpy()  # 掩码
                for j in range(0, ori_image.shape[0]):
                    # image_residual_n = One_mask_batch[j].reshape(256, 256)
                    # image_residual_th = residual_th(image_residual_n, threshold_per=0.05)
                    # image_residual_th = image_residual_th.reshape((256, 256))
                    # image_residual_th = (image_residual_th > 0).astype(np.uint8) * 255
                    # cv2.imwrite("pre_res.png", One_mask_batch[j])
                    image_residual_th = One_mask_batch[j].reshape((256, 256)) * 255
                    image_residual_th = image_residual_th.astype(np.uint8)
                    # cv2.imwrite("res.png", image_residual_th)
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_residual_th, connectivity=8)

                    for label in range(1, num_labels):  # 0 是背景，从 1 开始
                        area = stats[label, cv2.CC_STAT_AREA]
                        if area < 7000:  # 面积小于 500 的白色区域置为黑色
                            image_residual_th[labels == label] = 0

                    One_mask_batch[j] = image_residual_th / 255.0
                    # 对残差图像进行中值滤波；
                    # image_residual_th[j] = med_filt(image_residual_th, size=3)

                One_mask_batch = torch.from_numpy(One_mask_batch).cuda()
                # save_tensor_image(One_mask_batch, filenamemass)
                # 测试阶段，结合一阶段输出，拼接二阶段输入数据
                for j in range(0, ori_image.shape[0]):  # 一个batch
                    image_path = sample_batched["image_path"][0]
                    # 得到与测试图片最为相似的正常图片（tensor 格式）
                    similar_good_image = get_similar_image(image_path, obj_name)  # mvtec
                    # similar_good_image = get_similar_image_hash(data_excel, image_path, obj_name,image_test=ori_image[0]) #mvtec 3d
                    pre_mask = One_mask_batch[j]
                    similar_good_image = similar_good_image.cuda()
                    # augment_image[j] = torch.where(pre_mask > 0, ori_image[j], ori_image[j])
                    augment_image[j] = torch.where(pre_mask > 0, similar_good_image, ori_image[j])
                    image = np.array(augment_image[j].cpu()).astype(np.float32)  # 测一阶段+二阶段
                    image = np.transpose(image, (1, 2, 0))
                    imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
                    imagegray = imagegray[:, :, None]
                    imagegray = np.transpose(imagegray, (2, 0, 1))
                    input_image[j] = torch.tensor(imagegray).cuda()

                # save_tensor_image(augment_image, filenamemasss)
                if obj_name == 'bottle':
                    gradient = kornia.morphology.gradient(input_image, kernel)
                    gray_rec = model(gradient)  # 输入轮廓图
                else:
                    gray_rec = model(input_image)  # 输入灰度图
                # end_time = time.time()
                # execution_time = end_time - start_time
                # print("代码执行时间：", execution_time, "秒")
                # gray_rec = model(gray_gray)  # 只测二阶段
                # save_tensor_image(gray_rec, filenamemassss)
                recimg = gray_rec.detach().cpu().numpy()[0]  # 处理重构后的图片转为np
                recimg = np.transpose(recimg, (1, 2, 0)) * 180
                recimg = recimg.astype('uint8')
                # 处理 原始图片转为np
                oriimg = ori_image.detach().cpu().numpy()[0]
                oriimg = np.transpose(oriimg, (1, 2, 0)) * 180
                oriimg = oriimg.astype('uint8')
                # color
                colorD = ColorDifference(recimg, oriimg)
                # msgms
                mgsgmsmap = msgms(gray_rec, ori_image, as_loss=False)
                mgsgmsmapmean = mean_smoothing(mgsgmsmap, 21)
                out_mask_gradient = mgsgmsmapmean.detach().cpu().numpy()
                # combined
                out_mask_averaged = colorD[None, None, :, :] + out_mask_gradient


                saveimages = False
                # '''save result images
                if saveimages:
                    segresult = out_mask_averaged[0, 0, :, :]
                    truemaskresult = true_mask[0, 0, :, :]
                    # see_img(input_image,f'{savepath}/{obj_name}/',i,'rec')
                    # see_img(gray_gray,f'{savepath}/{obj_name}/',i,'orig')
                    # see_img_heatmap(gray_batch,segresult,f'{savepath}/{obj_name}/',i,'hetamap')
                    savefig(ori_image, segresult, truemaskresult, f'{savepath}/{obj_name}/' + f'segresult{i}.png',
                            gray_rec)
                    i = i + 1
                # '''

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
        # **************************
        # 输出预测结果
        # samples_info = list(zip(anomaly_score_gt, anomaly_score_prediction, range(len(anomaly_score_gt))))
        # samples_info_sorted = sorted(samples_info, key=lambda x: x[1])  # 按照概率值排序
        #
        # for true_label, pred_prob, sample_index in samples_info_sorted:
        #     print(f"Sample Index: {sample_index}, True Label: {true_label}, Predicted Probability: {pred_prob}")
        # **************************
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
    parser.add_argument('--checkpoint_path', action='store', type=str, default='E:/sjk/two_stage/Mod/btad/', required=False)
    parser.add_argument('--saveimages', default='True', action='store_true', )
    args = parser.parse_args()
    savepath = args.checkpoint_path

    obj_list = [
        'capsule',
        'bottle',
        'carpet',
        'leather',
        'pill',
        'transistor',
        'tile',
        'cable',
        'zipper',
        'toothbrush',
        'metal_nut',
        'hazelnut',
        'screw',
        'grid',
        'wood'
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

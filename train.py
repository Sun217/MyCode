import torch
from dataloader import MVTecDRAEMTrainDataset , MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
from torch import optim
from loss import SSIM
from test import get_similar_image
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unetskip import ReconstructiveSubNetwork
import cv2
from utils import savefig
from msgms import MSGMSLoss
import kornia
from ColorlossLab import ColorDifference
import torchvision.transforms.functional as TF
from torchvision import transforms
import copy
from PIL import Image
from seg_model import Seg_Network

toPIL = transforms.ToPILImage()


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


def mean_smoothing(amaps, kernel_size: int = 21):
    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
    mean_kernel = mean_kernel.to(amaps.device)
    return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)


def train_on_device(obj_names, args):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    for obj_name in obj_names:
        run_name = 'btad_' + str(args.lr) + '_' + str(args.epochs) + '_bs' + str(args.bs) + "_" + obj_name + '_'
        model = Seg_Network(in_channels=3, out_channels=1)
        model.cuda()
        model.apply(weights_init)
        optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.lr},], weight_decay=0)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args.epochs * 0.45, args.epochs * 0.9], gamma=0.2,last_epoch=-1)
        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM()
        img_dim = 256
        dataset = MVTecDRAEMTrainDataset(args.data_path + obj_name + "/train/good/", args.data_path + obj_name + "/train/good/", resize_shape=[256, 256],stage='one') # mvtec
        dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=0)

        Loss_max = 100
        aucm = 0
        for epoch in range(args.epochs):
            Loss = 0
            model.train()
            for i_batch, sample_batched in enumerate(dataloader):
                anomaly_mask = sample_batched["anomaly_mask"].cuda()
                augmented_image = sample_batched["augmented_image"].cuda()
                gray_rec = model(augmented_image)
                l2_loss = loss_l2(gray_rec, anomaly_mask)
                ssim_loss = loss_ssim(gray_rec, anomaly_mask)
                loss = l2_loss + ssim_loss
                Loss = loss + Loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if Loss < Loss_max:
                Loss_max = Loss
                torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name + 'best_1' + ".pckl"))
            else:
                torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name + 'last_1' + ".pckl"))
            print('obj_name:{:}  epoch:{:}  Loss:{:.4f}   Loss_max:{:.4f}'.format(obj_name, epoch, Loss, Loss_max))
            scheduler.step()

if __name__ == "__main__":
    if __name__ == "__main__":
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--obj_id', action='store', type=int, default='-1', required=False)
        parser.add_argument('--bs', action='store', type=int, default='4', required=False)
        parser.add_argument('--lr', action='store', type=float, default='0.0001', required=False)
        parser.add_argument('--epochs', action='store', type=int, default='200', required=False)
        parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
        parser.add_argument('--data_path', action='store', type=str, default='E:/btad_mvtec/', required=False)
        parser.add_argument('--anomaly_source_path', action='store', type=str, default='E:/btad_mvtec/',required=False)
        parser.add_argument('--checkpoint_path', action='store', type=str, default='.1/btad/', required=False)
        parser.add_argument('--log_path', action='store', type=str, default='E:/lxy/EdgRec6.4/Mod', required=False)
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
            # obj_list = [
            #     # 'bottle',
            #     # 'carpet',
            #     # 'leather',
            #     # 'pill',
            #     # 'screw',
            #     # 'capsule',
            #     # 'transistor',
            #     # 'tile',
            #     # 'cable',
            #     # 'zipper',
            #     # 'toothbrush',
            #     'metal_nut',
            #     # 'hazelnut',
            #     # 'grid',
            #     # 'wood'
            # ]
            # obj_list = [ # mvtec 3d
            #             # 'bagel',
            #             #  'cable_gland',
            #             #  'carrot',
            #              'cookie',
            #              'dowel',
            #             'foam',
            #             'peach',
            #             'potato',
            #             'rope',
            #             'tire',
            #             ]

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
            obj_list = [  # btad
                # '01',
                # '03',
                '02',
            ]
            picked_classes = obj_list
        else:
            picked_classes = obj_batch[int(args.obj_id)]

        with torch.cuda.device(args.gpu_id):
            train_on_device(picked_classes, args)

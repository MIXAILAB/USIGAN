
import sys
import os

# 获取当前脚本的绝对路径（calculate_ssim_folder.py）
current_file = os.path.abspath(__file__)
# 计算项目根目录的路径（假设目录结构为 project_root/scripts/metrics/）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
# 将项目根目录添加到 Python 的模块搜索路径
sys.path.insert(0, project_root)
import random
import csv
import torch
import argparse
import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor 
from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, help='Path to the folder.')
    args = parser.parse_args()
    return args


def calculate_psnr_ssim(fake_dirs,gt_dirs):
    pred_dir = fake_dirs
    targ_dir = gt_dirs
    img_list = [f for f in os.listdir(pred_dir) if f.endswith(('png', 'jpg'))]
    img_format = '.' + img_list[0].split('.')[-1]
    img_list = [f.replace('.png', '').replace('.jpg', '') for f in img_list]
    random.seed(0)
    random.shuffle(img_list)
    # PSNR and SSIM statistics
    psnr = []
    ssim = []
    
    for i in tqdm(img_list):
        fake = io.imread(os.path.join(pred_dir, i + img_format))
        real = io.imread(os.path.join(targ_dir, i + img_format))
        PSNR = peak_signal_noise_ratio(fake, real)
        psnr.append(PSNR)
        SSIM = structural_similarity(fake, real, multichannel=True,channel_axis=2)
        ssim.append(SSIM)
    average_psnr = sum(psnr)/len(psnr)
    average_ssim = sum(ssim)/len(ssim)
    print(pred_dir)
    print("The average psnr is " + str(average_psnr))
    print("The average ssim is " + str(average_ssim))
    print(f"{average_psnr:.4f} {average_ssim:.4f}")
    return average_psnr,average_ssim

def main(args):
    virtualstain_res_dir = os.path.join(args.dataroot,'fake')
    gt_stain_dir = os.path.join(args.dataroot,'gt')
    psnr,ssim = calculate_psnr_ssim(virtualstain_res_dir,gt_stain_dir)
    
    



if __name__ == "__main__":
    args = parse_args()
    main(args)
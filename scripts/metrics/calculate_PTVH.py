
import sys
import os

# 获取当前脚本的绝对路径（calculate_ssim_folder.py）
current_file = os.path.abspath(__file__)
# 计算项目根目录的路径（假设目录结构为 project_root/scripts/metrics/）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
# 将项目根目录添加到 Python 的模块搜索路径
sys.path.insert(0, project_root)

import os
import random
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

from scripts.metrics.perceptual import PerceptualHashValue




def calculate_PTVH_layers(fake_dirs,gt_dirs):
    pred_dir = fake_dirs
    targ_dir = gt_dirs
    img_list = [f for f in os.listdir(pred_dir) if f.endswith(('png', 'jpg'))]
    img_format = '.' + img_list[0].split('.')[-1]
    img_list = [f.replace('.png', '').replace('.jpg', '') for f in img_list]
    random.seed(0)
    random.shuffle(img_list)

    # PHV statistics
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    layers = ['layer_1', 'layer_2', 'layer_3', 'layer_4']
    PHV = PerceptualHashValue(
            T=0.01, network='resnet50', layers=layers, 
            resize=False, resize_mode='bilinear',
            instance_normalized=False).to(device)
    all_phv = []
    for i in tqdm(img_list):
        fake = io.imread(os.path.join(pred_dir, i + img_format))
        real = io.imread(os.path.join(targ_dir, i + img_format))

        fake = to_tensor(fake).to(device)
        real = to_tensor(real).to(device)

        phv_list = PHV(fake, real)
        all_phv.append(phv_list)
    all_phv = np.array(all_phv)
    all_phv = np.mean(all_phv, axis=0)
    res_str = ''
    for layer, value in zip(layers, all_phv):
        res_str += f'{layer}: {value:.4f} '
    print(res_str)
    print('PTVH layers:',np.round(all_phv, 4))
    return np.round(all_phv, 4) 
"""
测量的指标添加到experiments/evaluation/{$dataset_type}_metrics.csv
"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, help='Path to the folder.')
    parser.add_argument('--dataset_type', type=str, help='Which metric.csv to choose.')
    parser.add_argument('--exp_name', type=str, help='which exp name to store in csv')
    args = parser.parse_args()
    return args




def main(args):
    virtualstain_res_dir = os.path.join(args.dataroot,'fake')
    gt_stain_dir = os.path.join(args.dataroot,'gt')
    PTVHValues=calculate_PTVH_layers(virtualstain_res_dir,gt_stain_dir)
    # command = f'python3 src/metrics/kid_score.py --true {gt_stain_dir} --fake {virtualstain_res_dir}'
    # os.system(command)


if __name__ == "__main__":
    args = parse_args()
    main(args)

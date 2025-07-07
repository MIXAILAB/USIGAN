
import sys
import os

# 获取当前脚本的绝对路径（calculate_ssim_folder.py）
current_file = os.path.abspath(__file__)
# 计算项目根目录的路径（假设目录结构为 project_root/scripts/metrics/）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
# 将项目根目录添加到 Python 的模块搜索路径
sys.path.insert(0, project_root)
sys.path.insert(0, "./")
import csv
import torch
import argparse
from scripts.metrics.interact import set_logger
from scripts.metrics.fid import calculate_fid_given_paths
# from .utils.logger import log_to_csv
from scripts.metrics.calculate_fid_folder import calculate_fid
from scripts.metrics.calculate_PTVH import calculate_PTVH_layers
from scripts.metrics.calculate_ssim_folder import calculate_psnr_ssim


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

def evaluate_metrics(args):
    virtualstain_res_dir = os.path.join(args.dataroot,args.dataset_type,args.exp_name,'fake')
    gt_stain_dir = os.path.join(args.dataroot,args.dataset_type,args.exp_name,'gt')
    fidValue=calculate_fid(virtualstain_res_dir,gt_stain_dir)
    psnr,ssim = calculate_psnr_ssim(virtualstain_res_dir,gt_stain_dir)
    ptvhs = calculate_PTVH_layers(virtualstain_res_dir,gt_stain_dir)
    # save_path=os.path.join(project_root,'experiments','evaluation',f'{args.dataset_type}_metrics.csv')
    # log_to_csv(args.exp_name,fidValue,ptvhs,psnr,ssim,save_path)
    


if __name__ == "__main__":
    args = parse_args()
    evaluate_metrics(args)
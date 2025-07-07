import sys
import os

# 获取当前脚本的绝对路径（calculate_ssim_folder.py）
current_file = os.path.abspath(__file__)
# 计算项目根目录的路径（假设目录结构为 project_root/scripts/metrics/）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
# 将项目根目录添加到 Python 的模块搜索路径
sys.path.insert(0, project_root)

import csv
import torch
import argparse
from scripts.metrics.interact import set_logger
from scripts.metrics.fid import calculate_fid_given_paths
# from src.utils.logger import log_to_csv

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

def calculate_fid(translate_path,gt_path):
    path1 = translate_path
    dataset=""
    fid_value = calculate_fid_given_paths(paths=[path1, gt_path],dataset = dataset)
    print('fid:{}'.format(fid_value))
    return fid_value


def main(args):
    virtualstain_res_dir = os.path.join(args.dataroot,'fake')
    gt_stain_dir = os.path.join(args.dataroot,'gt')
    fidValue=calculate_fid(virtualstain_res_dir,gt_stain_dir)
    save_path=os.path.join(project_root,'experiments','evaluation',f'{args.dataset_type}_metrics.csv')
    # log_to_csv(args.exp_name,fidValue,save_path)
    # command = f'python3 src/metrics/kid_score.py --true {gt_stain_dir} --fake {virtualstain_res_dir}'
    # os.system(command)


if __name__ == "__main__":
    args = parse_args()
    main(args)
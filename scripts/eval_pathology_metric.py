"""
普通的图像指标难以完整描述生成结果

"""
import sys
import os

# 获取当前脚本的绝对路径（calculate_ssim_folder.py）
current_file = os.path.abspath(__file__)
# 计算项目根目录的路径（假设目录结构为 project_root/scripts/metrics/）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
# 将项目根目录添加到 Python 的模块搜索路径
sys.path.insert(0, project_root)

import cv2
import csv
import torch
import argparse
import numpy as np 
from scipy.stats import pearsonr 


"""
测量的指标添加到experiments/evaluation/{$dataset_type}_metrics.csv
"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, help='Path to the folder.')
    parser.add_argument('--expname', type=str, help='Path to the folder.')
    args = parser.parse_args()
    return args

def calculate_pearson_r_for_folders(generated_folder, gt_folder, resize_dim=None):
    """
    计算生成图片与 GT 图片的皮尔逊相关系数。
    
    参数:
        generated_folder (str): 生成结果图片文件夹路径。
        gt_folder (str): GT 图片文件夹路径。
        resize_dim (tuple): 调整图片大小的目标尺寸 (宽, 高)，默认为 None 表示不调整大小。
        
    返回:
        list: 包含每对图片的文件名和对应的皮尔逊相关系数。
    """
    # 获取文件夹中的文件名，并按字典序排序
    generated_files = sorted(os.listdir(generated_folder))
    gt_files = sorted(os.listdir(gt_folder))
    
    # 检查两个文件夹中的文件数量是否一致
    if len(generated_files) != len(gt_files):
        raise ValueError("生成的图片文件夹和 GT 文件夹中的图片数量不一致！")
    
    results = []
    
    # 遍历两个文件夹中对应的图片
    for gen_file, gt_file in zip(generated_files, gt_files):
        gen_path = os.path.join(generated_folder, gen_file)
        gt_path = os.path.join(gt_folder, gt_file)
        
        # 读取图片
        gen_image = cv2.imread(gen_path, cv2.IMREAD_GRAYSCALE)
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        # 检查是否成功读取图片
        if gen_image is None or gt_image is None:
            print(f"警告：无法读取图片 {gen_file} 或 {gt_file}，跳过...")
            continue
        
        # 调整图片大小（如果指定了 resize_dim）
        if resize_dim:
            gen_image = cv2.resize(gen_image, resize_dim)
            gt_image = cv2.resize(gt_image, resize_dim)
        
        # 将图片展开为一维数组
        gen_flat = gen_image.flatten()
        gt_flat = gt_image.flatten()
        
        # 计算皮尔逊相关系数
        corr, _ = pearsonr(gen_flat, gt_flat)
        print(f"GT:{gt_file} and Gen:{gen_file} correlation:",corr)
        # 保存结果
        results.append(corr)
        
    
    return results

if __name__ == "__main__":
    args = parse_args()
    gt_folder_path = os.path.join(args.dataroot,args.expname,"source")
    fake_folder_path = os.path.join(args.dataroot,args.expname,"fake")
    pearosn_r=calculate_pearson_r_for_folders(fake_folder_path,gt_folder_path)
    print("Pearson-R:",np.array(pearosn_r).mean())
import os.path
import numpy as np
import torch
import json

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util


class AlignedDataset(BaseDataset):
    """
    This dataset class can load aligned/paired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """
    """
    这个数据集类可以加载对齐/配对的数据集。

    它需要两个目录，分别用于存储来自领域A '/path/to/data/trainA' 和领域B '/path/to/data/trainB' 的训练图像。
    在训练时，可以使用 '--dataroot /path/to/data' 标志指定数据集。
    同样，在测试时，需要准备两个目录 '/path/to/data/testA' 和 '/path/to/data/testB'。
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        """初始化数据集类。

        参数:
            opt (Option类) -- 存储所有实验标志的类; 需要是BaseOptions的子类
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # 创建路径 '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # 创建路径 '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # 从 '/path/to/data/trainA' 加载图像
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # 从 '/path/to/data/trainB' 加载图像

        self.A_size = len(self.A_paths)  # 取数据集A的大小
        self.B_size = len(self.B_paths)  # 获取数据集B的大小
        assert self.A_size == self.B_size

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        """返回数据点及其元数据信息。

        参数:
            index (int)      -- 用于数据索引的随机整数

        返回包含A，B，A_paths和B_paths的字典
            A (tensor)       -- 输入域中的图像
            B (tensor)       -- 目标域中对应的图像
            A_paths (str)    -- 图像路径
            B_paths (str)    -- 图像路径
        """
        if self.opt.serial_batches:   # 确保索引在范围内
            index_B = index % self.B_size
        else:   # 随机化领域B的索引以避免固定对
            index = random.randint(0, self.A_size - 1)
            index_B = index % self.B_size
            
        A_path = self.A_paths[index]  # 确保索引在范围内
        B_path = self.B_paths[index_B]

        assert A_path == B_path.replace('trainB', 'trainA').replace('valB', 'valA').replace('testB', 'testA')

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # 应用图像转换
        # 对于CUT/FastCUT模式，在微调阶段（学习率下降），不执行CycleGAN的resize-crop数据增强。
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)

        # FDL: 同步转换
        seed = np.random.randint(2147483647) # 用numpy生成器生成种子 
        random.seed(seed) # 将此种子应用于图像转换
        torch.manual_seed(seed) # 适用于torchvision 0.7
        A = transform(A_img)
        random.seed(seed) # 将此种子应用于目标转换
        torch.manual_seed(seed) # 适用于torchvision 0.7
        B = transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """返回数据集中的图像总数。

        由于可能有两个具有不同图像数量的数据集，我们取最大值
        """
        return max(self.A_size, self.B_size)
"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod

#数据集的抽象基类（ABC）
class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).初始化类，首先调用 BaseDataset.__init__(self, opt)。
    -- <__len__>:                       return the size of dataset.返回数据集的大小
    -- <__getitem__>:                   get a data point.获取一个数据点
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.（可选）添加特定于数据集的选项并设置默认选项
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class初始化类；将选项保存在类中

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions   opt（Option 类）-- 存储所有实验标志；需要是 BaseOptions 的子类
        """
        self.opt = opt
        self.root = opt.dataroot
        self.current_epoch = 0

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.添加新的数据集特定选项，并重写现有选项的默认值。

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options. 是否为训练阶段。可以使用此标志添加特定于训练或测试的选项。

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""#返回数据集中的图像总数
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing 用于数据索引的随机整数

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information. 一个包含数据及其名称的字典。通常包含数据本身及其元数据信息
        """
        pass


def get_params(opt, size):
    # 获取输入图像的宽度和高度
    w, h = size 
    # 初始化新的高度和宽度为原始高度和宽度
    new_h = h 
    new_w = w
    # 根据预处理选项调整新的高度和宽度
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w
    # 随机生成裁剪的起始位置
    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))
    # 随机生成翻转的标志
    flip = random.random() > 0.5
    # 返回包含裁剪位置和翻转标志的字典
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    # 如果需要转换为灰度图像，添加灰度转换操作
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    # 根据预处理选项添加图像尺寸调整操作
    if 'fixsize' in opt.preprocess:
        transform_list.append(transforms.Resize(params["size"], method))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        if "gta2cityscapes" in opt.dataroot:
            osize[0] = opt.load_size // 2
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))
    elif 'scale_shortside' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_shortside(img, opt.load_size, opt.crop_size, method)))
    # 根据预处理选项添加随机缩放操作
    if 'zoom' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.Lambda(lambda img: __random_zoom(img, opt.load_size, opt.crop_size, method)))
        else:
            transform_list.append(transforms.Lambda(lambda img: __random_zoom(img, opt.load_size, opt.crop_size, method, factor=params["scale_factor"])))
    # 根据预处理选项添加裁剪操作
    if 'crop' in opt.preprocess:
        if params is None or 'crop_pos' not in params:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))
    # 根据预处理选项添加图像补丁操作
    if 'patch' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __patch(img, params['patch_index'], opt.crop_size)))
    # 根据预处理选项添加图像修剪操作
    if 'trim' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __trim(img, opt.crop_size)))
    # 如果预处理选项为'none'，添加使图像边长变为2的幂次方的操作
    # if opt.preprocess == 'none':
    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))
    # 如果不禁用翻转，根据参数添加水平翻转操作
    if not opt.no_flip:
        if params is None or 'flip' not in params:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif 'flip' in params:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    # 如果需要将图像转换为张量，添加ToTensor和归一化操作
    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # 返回组合的图像变换操作
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    # 获取原始图像的宽度和高度
    ow, oh = img.size
    # 将图像的高度和宽度调整为给定基数的幂次方
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    # 如果调整后的高度和宽度与原始图像相同，直接返回原始图像
    if h == oh and w == ow:
        return img
    # 使用指定的方法对图像进行调整大小
    return img.resize((w, h), method)


def __random_zoom(img, target_width, crop_width, method=Image.BICUBIC, factor=None):
    # 如果没有指定缩放因子，随机生成缩放比例
    if factor is None:
        zoom_level = np.random.uniform(0.8, 1.0, size=[2])
    else:
    # 使用指定的缩放因子
        zoom_level = (factor[0], factor[1])
    # 获取原始图像的宽度和高度
    iw, ih = img.size
    # 计算缩放后的宽度和高度，确保至少为指定的裁剪宽度
    zoomw = max(crop_width, iw * zoom_level[0])
    zoomh = max(crop_width, ih * zoom_level[1])
    # 使用指定的方法对图像进行缩放
    img = img.resize((int(round(zoomw)), int(round(zoomh))), method)
    # 返回缩放后的图像
    return img


def __scale_shortside(img, target_width, crop_width, method=Image.BICUBIC):
    # 获取原始图像的宽度和高度
    ow, oh = img.size
    # 获取图像短边的长度
    shortside = min(ow, oh)
    # 如果短边长度大于等于目标宽度，直接返回原始图像
    if shortside >= target_width:
        return img
    else:
        # 计算缩放比例，使得短边长度等于目标宽度
        scale = target_width / shortside
        # 使用指定的方法对图像进行缩放
        return img.resize((round(ow * scale), round(oh * scale)), method)


def __trim(img, trim_width):
    # 获取原始图像的宽度和高度
    ow, oh = img.size
    # 如果宽度大于指定的修剪宽度，随机选择裁剪区域
    if ow > trim_width:
        xstart = np.random.randint(ow - trim_width)
        xend = xstart + trim_width
    else:
        # 如果宽度小于等于修剪宽度，不进行裁剪
        xstart = 0
        xend = ow
    # 如果高度大于指定的修剪宽度，随机选择裁剪区域
    if oh > trim_width:
        ystart = np.random.randint(oh - trim_width)
        yend = ystart + trim_width
    else:
        # 如果高度小于等于修剪宽度，不进行裁剪
        ystart = 0
        yend = oh
    # 返回裁剪后的图像
    return img.crop((xstart, ystart, xend, yend))


def __scale_width(img, target_width, crop_width, method=Image.BICUBIC):
    # 获取原始图像的宽度和高度
    ow, oh = img.size
    # 如果宽度已经等于目标宽度且高度大于等于裁剪宽度，直接返回原始图像
    if ow == target_width and oh >= crop_width:
        return img
    # 计算等比例缩放后的宽度和高度
    w = target_width
    h = int(max(target_width * oh / ow, crop_width))
    # 使用指定的方法对图像进行缩放
    return img.resize((w, h), method)


def __crop(img, pos, size):
    # 获取原始图像的宽度和高度
    ow, oh = img.size
    # 获取裁剪的起始位置和裁剪尺寸
    x1, y1 = pos
    tw = th = size
    # 如果图像宽度或高度大于裁剪尺寸，进行裁剪
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    # 如果图像宽度和高度都不大于裁剪尺寸，直接返回原始图像
    return img


def __patch(img, index, size):
    # 获取原始图像的宽度和高度
    ow, oh = img.size
    # 计算网格数量
    nw, nh = ow // size, oh // size
    # 计算在宽度和高度上的余量
    roomx = ow - nw * size
    roomy = oh - nh * size
    # 在余量内随机选择裁剪起始位置
    startx = np.random.randint(int(roomx) + 1)
    starty = np.random.randint(int(roomy) + 1)
    # 根据给定的索引计算裁剪位置
    index = index % (nw * nh)
    ix = index // nh
    iy = index % nh
    gridx = startx + ix * size
    gridy = starty + iy * size
    # 返回裁剪后的图像
    return img.crop((gridx, gridy, gridx + size, gridy + size))


def __flip(img, flip):
    # 如果需要翻转，左右翻转图像
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    # 否则直接返回原始图像
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)""""""打印关于图像大小的警告信息（仅打印一次）"""
    if not hasattr(__print_size_warning, 'has_printed'):
        #print("图像大小应为4的倍数。加载的图像大小为 (%d, %d)，因此已调整为 "
        #     "(%d, %d)。此调整将应用于所有大小不是4的倍数的图像。" % (ow, oh, w, h))
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True

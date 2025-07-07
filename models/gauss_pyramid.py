import torch
from torch import nn

class Gauss_Pyramid_Conv(nn.Module):
    """
    Code borrowed from: https://github.com/csjliang/LPTN
    高斯金字塔卷积模块
    """
    def __init__(self, num_high=3):
        super(Gauss_Pyramid_Conv, self).__init__()
        # 设置高斯金字塔层数
        self.num_high = num_high
        # 生成高斯核
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda'), channels=3):
        # 定义5x5的高斯核
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        # 归一化高斯核
        kernel /= 256.
        # 复制高斯核到指定通道数
        kernel = kernel.repeat(channels, 1, 1, 1)
        # 将高斯核移动到指定设备
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        # 对输入进行下采样（每隔一个像素取一个）
        return x[:, :, ::2, ::2]

    def conv_gauss(self, img, kernel):
        # 对图像进行反射填充
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        # 使用分组卷积进行卷积操作
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def forward(self, img):
        # 初始化当前层为输入图像
        current = img
        pyr = []
        for _ in range(self.num_high):
            # 对当前层进行高斯卷积
            filtered = self.conv_gauss(current, self.kernel)
            # 将卷积结果添加到金字塔列表
            pyr.append(filtered)
            # 对卷积结果进行下采样
            down = self.downsample(filtered)
            # 更新当前层为下采样结果
            current = down
        # 将最后一层（下采样后的最小分辨率）加入金字塔列表
        pyr.append(current)
        return pyr

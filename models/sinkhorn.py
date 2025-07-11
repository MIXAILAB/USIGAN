# -*- coding: utf-8 -*-
from packaging import version
import torch


def sinkhorn(dot, max_iter=100):
    """使用Sinkhorn算法对输入的dot进行归一化处理
    dot: n x in_size x out_size
    mask: n x in_size
    output: n x in_size x out_size
    """
    n, in_size, out_size = dot.shape
    K = dot
    # K: n x in_size x out_size
    u = K.new_ones((n, in_size))# 初始化u矩阵，形状为 n x in_size，元素为1
    v = K.new_ones((n, out_size))# 初始化v矩阵，形状为 n x out_size，元素为1
    a = float(out_size / in_size) # 计算常数a，用于归一化
    for _ in range(max_iter):
        u = a / torch.bmm(K, v.view(n, out_size, 1)).view(n, in_size)# 更新u矩阵
        v = 1. / torch.bmm(u.view(n, 1, in_size), K).view(n, out_size)# 更新v矩阵
    K = u.view(n, in_size, 1) * (K * v.view(n, 1, out_size))# 根据u和v更新K矩阵
    return K

def OT(q, k, eps=1.0, max_iter=100, cost_type=None):
    """使用Sinkhorn OT算法计算权重
    q: n x in_size x in_dim
    k: m x out_size x in_dim (m: number of heads/ref)
    output: n x out_size x m x in_size
    """
    n, in_size, in_dim = q.shape
    m, out_size = k.shape[:-1]
    # 计算输入的C矩阵，形状为 n x m x in_size x out_size
    C = torch.einsum('bid,bod->bio', q, k) # 计算相似度矩阵
    if cost_type == 'easy':
        K = 1 - C.clone()
    elif cost_type == 'hard':
        K = C.clone()
    #K = 1 - C.clone()
    npatches = q.size(1)
    # 创建对角矩阵，并在K中将对角线元素置为-10
    mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
    diagonal = torch.eye(npatches, device=q.device, dtype=mask_dtype)[None, :, :]
    K.masked_fill_(diagonal, -10)

    # 将K重新调整形状为 nm x in_size x out_size
    # K: n x m x in_size x out_size
    K = K.reshape(-1, in_size, out_size)
    # K: nm x in_size x out_size
    # 对K进行指数运算，然后使用Sinkhorn算法进行归一化处理
    K = torch.exp(K / eps)
    K = sinkhorn(K, max_iter=max_iter)
    # K: nm x in_size x out_size
    K = K.permute(0, 2, 1).contiguous()
    #print("K的shape：",K.shape)
    return K
    



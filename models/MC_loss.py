from packaging import version
import torch
from torch import nn
import math
from .sinkhorn import OT
import numpy as np
import torch.nn.functional as F
from .uot_loss import UOT


class MC_Loss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        #self.l2_norm = Normalize(2)

    def forward(self, feat_src, feat_tgt, feat_gen):
        batchSize = feat_src.shape[0]
        dim = feat_src.shape[1]   
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size // len(self.opt.gpu_ids)

        # if self.loss_type == 'MoNCE':
        ot_src = feat_src.view(batch_dim_for_bmm, -1, dim).detach()
        ot_tgt = feat_tgt.view(batch_dim_for_bmm, -1, dim).detach()
        ot_gen = feat_gen.view(batch_dim_for_bmm, -1, dim)
        # print("ot_src:",ot_src.shape) Bx xdim
        # print("ot_tgt:",ot_tgt.shape)
        # print("ot_gen:",ot_gen.shape)
        f1 = OT(ot_src, ot_tgt, eps=self.opt.eps, max_iter=50, cost_type=self.opt.cost_type)
        #print("F1:",f1)
        f2 = OT(ot_tgt, ot_gen, eps=self.opt.eps, max_iter=50, cost_type=self.opt.cost_type)
        #print("F2:",f2)
        MC_Loss = F.l1_loss(f1, f2) 
        return   MC_Loss


class UMC_Loss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        #self.l2_norm = Normalize(2)
        self.p,self.blur = 1,0.025

    def forward(self, feat_src, feat_tgt, feat_gen):
        batchSize = feat_src.shape[0]
        dim = feat_src.shape[1]   
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size // len(self.opt.gpu_ids)

        # if self.loss_type == 'MoNCE':
        ot_src = feat_src.view(batch_dim_for_bmm, -1, dim).detach()
        ot_tgt = feat_tgt.view(batch_dim_for_bmm, -1, dim).detach()
        ot_gen = feat_gen.view(batch_dim_for_bmm, -1, dim)
        # print("ot_src:",ot_src.shape) # 2x256x256
        #print("ot_tgt:",ot_tgt.shape)
        #print("ot_gen:",ot_gen.shape)
        f1 = UOT(ot_src, ot_tgt,tau=0.001, eps=self.opt.eps, max_iter=50, cost_type=self.opt.cost_type)
        # print("F1:",f1.shape)
        f2 = UOT(ot_tgt, ot_gen,tau=0.001, eps=self.opt.eps, max_iter=50, cost_type=self.opt.cost_type)
        # print("F2:",f2.shape) # 2x256x256
        f_indirect = torch.matmul(f1, f2)  # [B, HW_src, HW_gen]
        f_direct = UOT(ot_src, ot_gen, tau=0.001, eps=self.opt.eps, max_iter=50, cost_type=self.opt.cost_type)  # [B, HW_src, HW_gen]
        cyc_loss = F.l1_loss(f_direct, f_indirect)
        return cyc_loss

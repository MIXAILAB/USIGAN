
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from PIL import Image
import math

# consistency ot
# Pathological Corresponse Self-Mining
class PCSM_Loss(nn.Module):
    def __init__(self,device):
        super(PCSM_Loss,self).__init__()
        # tradional color deconvolution for stain separation
        self.rgb_from_hed = torch.tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]]).to(device)
        self.coeffs = torch.tensor([0.2125, 0.7154, 0.0721]).view(3,1).to(device)
        self.hed_from_rgb = torch.linalg.inv(self.rgb_from_hed).to(device)
        # focal FOD , alpha 
        self.alpha = 1.8 
        self.adjust_Calibration = torch.tensor(10**(-(math.e)**(1/self.alpha))).to(device) # 
        
        # Set a threshold to identify and zero out FOD values that are too low,/
        # thereby reducing their impact on computing the tumor expression level inference.
        self.thresh_FOD = 0.15
        # thresh_FOD for getting pseudo mask
        self.thresh_mask = 0.68

        self.log_adjust = torch.log(torch.tensor(1e-6)).to(device)
        self.device = device
        self.mse_loss = nn.MSELoss().to(device)
    
    def forward(self,src,tgt,gen,src_feats,tgt_feats,gen_feats):
        input_reshape = tgt.permute(0,2,3,1)
        tgt_reshape = gen.permute(0,2,3,1)
        inputs_OD,input_avg_OD,input_msk = self.compute_OD(input_reshape)
        output_OD,output_avg_OD,output_msk = self.compute_OD(tgt_reshape)
        # print(inputs_OD.shape) BxHxW
        # masked_src = src*input_msk.unsqueeze(1)
        # normalize aviod big number
        src_OD_maps = inputs_OD.unsqueeze(1)#(inputs_OD-torch.min(inputs_OD))/(torch.max(inputs_OD)-torch.min(inputs_OD))
        tgt_OD_maps = output_OD.unsqueeze(1)#(output_OD-torch.min(output_OD))/(torch.max(output_OD)-torch.min(output_OD))
        src_matrix= self.cal_matrix([src_OD_maps]).detach().to(self.device)
        tgt_matrix= self.cal_matrix([tgt_OD_maps]).to(self.device)
        
        loss = F.l1_loss(src_matrix,tgt_matrix)
        # print(loss)
        # print((self.mse_loss(input_avg_OD.to(self.device),output_avg_OD)/(src.shape[2]*src.shape[3])**2))
        loss += (self.mse_loss(input_avg_OD.to(self.device),output_avg_OD)/(src.shape[2]*src.shape[3])**2) # todo: 是否起作用？
        # src_scm_matrix = self.cal_self_corresponse_matrix(src_feats[-1])
        # tgt_scm_matrix = self.cal_self_corresponse_matrix(gen_feats[-1])
        # loss += F.l1_loss(src_scm_matrix,tgt_scm_matrix)#*10
        
        # print(loss1,loss2)
        
        return loss
        
    def cal_matrix(self,feats_pool):
        batch_images = feats_pool[0]
        cosine_similarity_matrices = torch.zeros(len(feats_pool), feats_pool[0].size(0), feats_pool[0].size(0))
        # 遍历每个四维张量
        for idx, tensor in enumerate(feats_pool):
            # print(tensor.shape)
    # 切分为8个四维张量，第一维度为1
            sub_tensors = torch.split(tensor, 1, dim=0)

    # 计算任意两个张量之间的余弦相似度
            for i in range(batch_images.size(0)):
                for j in range(batch_images.size(0)):
                    vector_i = sub_tensors[i].view(-1)
                    vector_j = sub_tensors[j].view(-1)

            # 使用余弦相似度公式计算
                    similarity = F.cosine_similarity(vector_i, vector_j, dim=0)
                    cosine_similarity_matrices[idx, i, j] = similarity
        return cosine_similarity_matrices
    
    def cal_self_corresponse_matrix(self,feat):
        B,C,H,W = feat.size() # only use high-level feat
        feature_flat = feat.view(B, C, H * W)        # [B, C, HW]
        feature_flat_t = feature_flat.permute(0, 2, 1)  # [B, HW, C]
        scm = torch.bmm(feature_flat_t, feature_flat)   # [B, HW, HW]
        scm = scm / (C ** 0.5)  # normalization
        return scm

    
    
    def compute_OD(self,image):
        assert image.shape[-1] == 3
        # Focal Optical Density map
        ihc_hed = self.separate_stains(image,self.hed_from_rgb)
        null = torch.zeros_like(ihc_hed[:,:, :, 0])
        # select DAB stain OD and generate RGB image only with DAB OD
        ihc_d = self.combine_stains(torch.stack((null, null, ihc_hed[:,:, :, 2]), axis=-1),self.rgb_from_hed)
        # turn into gray
        grey_d = self.rgb2gray(ihc_d)
        grey_d[grey_d<0.0] = torch.tensor(0.0).cuda()
        grey_d[grey_d>1.0] = torch.tensor(1.0).cuda()
        # get FOD in later process
        FOD = torch.log10(1/(grey_d+self.adjust_Calibration))
        FOD[FOD<0] = torch.tensor(0.0).cuda()
        FOD = FOD**self.alpha
        # Set a threshold to identify and zero out FOD values that are too low
        FOD_relu = torch.where(FOD < self.thresh_FOD, torch.tensor(0.0).cuda(), FOD)
        # mask_OD generate a pseudo mask for IHC image(real or fake)
        mask_OD = torch.where(FOD < self.thresh_mask, torch.tensor(0.0).cuda(), FOD)
        mask_OD = mask_OD.squeeze(-1).detach()
        mask_OD[mask_OD > 0] = torch.tensor(1.0)
        
        # flattened_img = FOD_relu.squeeze(-1).flatten(1,2)
        flattened_img_2 = FOD.flatten(1,2)
        
        # avg
        avg = torch.sum(FOD_relu,dim=(1,2,3))
        
        return flattened_img_2,avg, mask_OD
    
    def separate_stains(self,rgb, conv_matrix, *, channel_axis=-1):
        rgb = torch.maximum(rgb, torch.tensor(1e-6))  # avoiding log artifacts
        stains = torch.matmul(torch.log(rgb) / self.log_adjust, conv_matrix)
        stains = torch.maximum(stains, torch.tensor(0))

        return stains
    def combine_stains(self,stains, conv_matrix, *, channel_axis=-1):
        log_rgb = -torch.matmul((stains * -self.log_adjust) , conv_matrix)
        rgb = torch.exp(log_rgb)

        return torch.clamp(rgb, min=0, max=1)
    def rgb2gray(self,rgb, *, channel_axis=-1):
        return torch.matmul(rgb ,self.coeffs)
    
    

class ComputeODModule(nn.Module):
    def __init__(self,device):
        super(ComputeODModule,self).__init__()
        # tradional color deconvolution for stain separation
        self.rgb_from_hed = torch.tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]]).to(device)
        self.coeffs = torch.tensor([0.2125, 0.7154, 0.0721]).view(3,1).to(device)
        self.hed_from_rgb = torch.linalg.inv(self.rgb_from_hed).to(device)
        # focal FOD , alpha 
        self.alpha = 1.8 
        self.adjust_Calibration = torch.tensor(10**(-(math.e)**(1/self.alpha))).to(device) # 
        
        # Set a threshold to identify and zero out FOD values that are too low,/
        # thereby reducing their impact on computing the tumor expression level inference.
        self.thresh_FOD = 0.15
        # thresh_FOD for getting pseudo mask
        self.thresh_mask = 0.68

        self.log_adjust = torch.log(torch.tensor(1e-6)).to(device)
        self.device = device
        self.mse_loss = nn.MSELoss().to(device)
    
    def forward(self,src):
        input_reshape = src.permute(0,2,3,1)
       
        inputs_OD,input_avg_OD,input_msk = self.compute_OD(input_reshape)
       
        return inputs_OD,input_avg_OD,input_msk
        # print(inputs_OD.shape) BxHxW
        # masked_src = src*input_msk.unsqueeze(1)
        # normalize aviod big number
        # src_OD_maps = inputs_OD.unsqueeze(1)#(inputs_OD-torch.min(inputs_OD))/(torch.max(inputs_OD)-torch.min(inputs_OD))
        # tgt_OD_maps = output_OD.unsqueeze(1)#(output_OD-torch.min(output_OD))/(torch.max(output_OD)-torch.min(output_OD))
        # src_matrix= self.cal_matrix([src_OD_maps]).detach().to(self.device)
        # tgt_matrix= self.cal_matrix([tgt_OD_maps]).to(self.device)
        
        # loss = F.l1_loss(src_matrix,tgt_matrix)
        # # print(loss)
        # # print((self.mse_loss(input_avg_OD.to(self.device),output_avg_OD)/(src.shape[2]*src.shape[3])**2))
        # loss += (self.mse_loss(input_avg_OD.to(self.device),output_avg_OD)/(src.shape[2]*src.shape[3])**2)
        # src_scm_matrix = self.cal_self_corresponse_matrix(src_feats[-1])
        # tgt_scm_matrix = self.cal_self_corresponse_matrix(tgt_feats[-1])
        
        # loss += F.l1_loss(src_scm_matrix,tgt_scm_matrix)
        # # print(loss1,loss2)
        
        # return loss
        
    
    def compute_OD(self,image):
        assert image.shape[-1] == 3
        # Focal Optical Density map
        ihc_hed = self.separate_stains(image,self.hed_from_rgb)
        null = torch.zeros_like(ihc_hed[:,:, :, 0])
        # select DAB stain OD and generate RGB image only with DAB OD
        ihc_d = self.combine_stains(torch.stack((null, null, ihc_hed[:,:, :, 2]), axis=-1),self.rgb_from_hed)
        # turn into gray
        grey_d = self.rgb2gray(ihc_d)
        grey_d[grey_d<0.0] = torch.tensor(0.0).cuda()
        grey_d[grey_d>1.0] = torch.tensor(1.0).cuda()
        # get FOD in later process
        FOD = torch.log10(1/(grey_d+self.adjust_Calibration))
        FOD[FOD<0] = torch.tensor(0.0).cuda()
        FOD = FOD**self.alpha
        # Set a threshold to identify and zero out FOD values that are too low
        FOD_relu = torch.where(FOD < self.thresh_FOD, torch.tensor(0.0).cuda(), FOD)
        # mask_OD generate a pseudo mask for IHC image(real or fake)
        mask_OD = torch.where(FOD < self.thresh_mask, torch.tensor(0.0).cuda(), FOD)
        mask_OD = mask_OD.squeeze(-1).detach()
        mask_OD[mask_OD > 0] = torch.tensor(1.0)
        
        # flattened_img = FOD_relu.squeeze(-1).flatten(1,2)
        flattened_img_2 = FOD.flatten(1,2)
        
        # avg
        avg = torch.sum(FOD_relu,dim=(1,2,3))
        
        return FOD.permute(0,3,2,1),avg, mask_OD,grey_d
    
    def separate_stains(self,rgb, conv_matrix, *, channel_axis=-1):
        rgb = torch.maximum(rgb, torch.tensor(1e-6))  # avoiding log artifacts
        stains = torch.matmul(torch.log(rgb) / self.log_adjust, conv_matrix)
        stains = torch.maximum(stains, torch.tensor(0))

        return stains
    def combine_stains(self,stains, conv_matrix, *, channel_axis=-1):
        log_rgb = -torch.matmul((stains * -self.log_adjust) , conv_matrix)
        rgb = torch.exp(log_rgb)

        return torch.clamp(rgb, min=0, max=1)
    def rgb2gray(self,rgb, *, channel_axis=-1):
        return torch.matmul(rgb ,self.coeffs)
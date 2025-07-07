from packaging import version
import torch
from torch import nn


class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()
        # print("原始feat_q",feat_q.shape) # 
        # print("原始feat_k",feat_k.shape)
        # 计算正样本的 logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        if self.opt.nce_includes_all_negatives_from_minibatch:
            # 将特征 reshape 为 minibatch 大小为 1 时的负样本
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        #print("feat_q",feat_q.shape)
        #print("feat_k",feat_k.shape)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))
        #print("l_neg_curbatch",l_neg_curbatch.shape)

        # 对角线元素是相同特征之间的相似性，因此是无意义的。
        # 用非常小的数字填充对角线，即 exp(-10)，几乎为零
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        #print("diagonal",diagonal.shape)
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        #print("l_neg_curbatch",l_neg_curbatch.shape)
        l_neg = l_neg_curbatch.view(-1, npatches)
        #print("l_neg",l_neg.shape)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T
        #print("out",out.shape)
        # 计算交叉熵损失
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        #print("loss",loss.shape)
        return loss


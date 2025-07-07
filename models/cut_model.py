import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import torch.nn.functional as F
from .MC_loss import MC_Loss,UMC_Loss,UMC_LossV2
from .PCM_loss import PCSM_Loss,ComputeODModule
import sys
import torch.nn as nn    

class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """ 为CUT模型配置特定选项
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='GAN损失的权重：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='NCE损失的权重：NCE(G(X), X)')
        parser.add_argument('--lambda_CC', type=float, default=1.0, help='CC损失的权重：NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='对身份映射使用NCE损失：NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='在哪些层计算NCE损失')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='（用于单图翻译）如果为True，则在计算对比损失时包括minibatch的其他样本的负样本。请参见models/patchnce.py以获取更多详细信息。')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='如何下采样特征图')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='NCE损失的温度')
        parser.add_argument('--num_patches', type=int, default=256, help='每层的补丁数量')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="将翻转等变性作为附加正则化。FastCUT使用，但CUT不使用")

        parser.set_defaults(pool_size=0)  # 不使用图像池

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # 指定要打印的训练损失
        # 训练/测试脚本将调用<BaseModel.get_current_losses>
        #self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.loss_names = ['adv', 'nce','mc','odc','cc'] #'mc', 'cc' ,'mc','cc' 'pcscm','cc' # ,'odc','cc','mc'
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        self.pcsm_loss = PCSM_Loss(device=self.device)
        
        if opt.nce_idt and self.isTrain:
            #self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # 在测试时，只加载G
            self.model_names = ['G']

        # 定义网络（生成器和判别器）
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        # self.netF_ = networks.define_F(opt.input_nc, "sample", opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        
        self.getod = ComputeODModule(self.device)
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

             # 定义损失函数
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []
            self.criterionOT = []
  
            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
            for nce_layer in self.nce_layers:
                self.criterionOT.append(UMC_Loss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        # self-correlation map 
        # self.scmap = SelfCorrelationMap().to(self.device)

        
        
    def data_dependent_initialize(self, data):
        """
        特征网络netF是以netG编码器部分的中间提取特征的形状为基础定义的。
        因此，netF的权重在第一次前向传递时使用一些输入图像进行初始化。
        请还参见PatchSampleF.create_mlp()，该函数在第一次forward()调用时被调用。
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                    # 计算假图像：G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                 # 计算D的梯度
            self.compute_G_loss().backward()                  # 计算G的梯度
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # 前向传递
        self.forward()

        # 更新D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # 更新G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        """从数据加载器中解压输入数据并执行必要的预处理步骤。
        参数：
            input (dict)：包含数据本身及其元数据信息。
        选项'direction'可用于交换域A和域B。
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """运行前向传递；由<optimize_parameters>和<test>函数调用。"""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

        
    def compute_D_loss(self):
        """计算判别器的GAN损失"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        # 伪造；通过将fake_B分离，停止对生成器的反向传播
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # 结合损失并计算梯度
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """计算生成器的GAN和NCE损失"""
        fake = self.fake_B
        # 首先，G(A)应该欺骗判别器
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE
        CC_loss = self.calculate_CC_loss(self.real_B, self.fake_B)
        OT_loss = self.calculate_OT_loss(self.real_A, self.real_B, self.fake_B)
        feat_tgt = self.netG(self.real_B, self.nce_layers, encode_only=True)
        feat_gen = self.netG(self.fake_B, self.nce_layers, encode_only=True)
        feat_src = self.netG(self.real_A, self.nce_layers, encode_only=True)
        odc_loss = self.pcsm_loss(self.real_A,self.real_B,self.fake_B,feat_src,feat_tgt,feat_gen)
        self.loss_G = self.loss_G_GAN + loss_NCE_both  + OT_loss +CC_loss + odc_loss  
        self.loss_adv = self.loss_G_GAN + self.loss_D
        self.loss_nce = self.loss_NCE + self.loss_NCE_Y
        self.loss_odc = odc_loss
        self.loss_mc = OT_loss
        self.loss_cc = CC_loss

        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)
        
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE

            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
    
    def calculate_OT_loss(self, src, tgt, gen):
        n_layers = len(self.nce_layers)
        feat_src = self.netG(src, self.nce_layers, encode_only=True)
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_src = [torch.flip(fq, [3]) for fq in feat_src_layers]
        feat_tgt = self.netG(tgt, self.nce_layers, encode_only=True)
        feat_gen = self.netG(gen, self.nce_layers, encode_only=True)
        
        feat_src_pool, sample_ids = self.netF(feat_src, self.opt.num_patches, None)
        feat_tgt_pool, _ = self.netF(feat_tgt, self.opt.num_patches, sample_ids)
        feat_gen_pool, _ = self.netF(feat_gen, self.opt.num_patches, sample_ids)
        total_ot_loss = 0.0
        for f_src, f_tgt, f_gen, crit, nce_layer in zip(feat_src_pool, feat_tgt_pool, feat_gen_pool, self.criterionOT, self.nce_layers):
            
            loss = crit(f_src, f_tgt, f_gen) * 10000
            total_ot_loss += loss.mean()

        return total_ot_loss / n_layers

    
    def calculate_CC_loss(self, src, tgt):
        matrix_src = self.cal_matrix(src).detach().to(self.device)
        matrix_tgt = self.cal_matrix(tgt).to(self.device)
        CC_loss = F.l1_loss(matrix_src, matrix_tgt) * 10
        return CC_loss      
      
    
    def cal_matrix(self, batch_images):
        cosine_similarity_matrices = torch.zeros(len(self.nce_layers), batch_images.size(0), batch_images.size(0))
        feat = self.netG(batch_images, self.nce_layers, encode_only=True)

        # 遍历每个四维张量
        for idx, tensor in enumerate(feat):
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

    



    

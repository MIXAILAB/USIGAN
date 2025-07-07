from .tmux_launcher import Options, TmuxLauncher

# 定义 Launcher 类，继承自 TmuxLauncher
class Launcher(TmuxLauncher):
    # 定义 common_options 方法，返回一组默认的实验配置选项
    def common_options(self):
        return [
            # Command 0
            Options(
                dataroot="/root/Desktop/data/private/Dataset4Research/IHC4BC_data/ER",#ER/TrainValAB  #/root/Desktop/data/private/Dataset4Research/IHC4BC_data/PR
                name="ihc4bc_er",
                checkpoints_dir='USIGAN',
                model='cut',
                CUT_mode="CUT",
                
                n_epochs=40,  # number of epochs with the initial learning rate # 初始学习率的代数
                n_epochs_decay=40,  # number of epochs to linearly decay learning rate to zero # 学习率线性衰减至零的代数

                netD='n_layers',  # ['basic', 'n_layers, 'pixel', 'patch'], 'specify discriminator architecture. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
                # 判别器架构选择 ['basic', 'n_layers, 'pixel', 'patch']，'basic' 是一个 70x70 的 PatchGAN
                ndf=32,
                netG='resnet_6blocks',  # 生成器架构选择 ['resnet_9blocks', 'resnet_6blocks', 'unet_256', 'unet_128', 'stylegan2', 'smallstylegan2', 'resnet_cat'], 'specify generator architecture')
                n_layers_D=5,  # 'only used if netD==n_layers' # 只在 netD==n_layers 时使用
                normG='instance',  # ['instance, 'batch, 'none'], 'instance normalization or batch normalization for G')生成器 G 的归一化方式选择 
                normD='instance',  # ['instance, 'batch, 'none'], 'instance normalization or batch normalization for D')# 判别器 D 的归一化方式选择
                weight_norm='spectral',

                lambda_GAN=1.0,  # weight for GAN loss：GAN(G(X))# GAN 损失的权重：GAN(G(X))
                lambda_NCE=1.0,  # weight for NCE loss: NCE(G(X), X)# NCE 损失的权重：NCE(G(X), X)
                lambda_CC=10.0,
                nce_layers='0,4,8,12,16',
                nce_T=0.07,
                num_patches=256,
                epoch_count=1,
                # FDL:
                #lambda_gp=10.0,
                #gp_weights='[0.015625,0.03125,0.0625,0.125,0.25,1.0]',
                #lambda_asp=10.0,  # weight for NCE loss: NCE(G(X), X)# NCE 损失的权重：NCE(G(X), X)
                #asp_loss_mode='lambda_linear',
                # lr=0.0001,
                dataset_mode='aligned',  #数据集加载方式选择 chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
                direction='AtoB',
                # serial_batches='', # if true, takes images in order to make batches, otherwise takes them randomly
                num_threads=30,  # '# threads for loading data')# 数据加载的线程数
                batch_size=2,  # 'input batch size')# 输入批量大小
                gpu_ids="0",
                load_size=1024,  # 'scale images to this size')# 缩放图像到这个大小
                crop_size=512,  # 'then crop to this size')# 裁剪图像到这个大小  # when test/eval change it to 1024
                preprocess='resize_and_crop',  # # 图像加载时的预处理方式选择='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
                # no_flip='',
                flip_equivariance=False,
                display_winsize=1024,  # display window size for both visdom and HTML  visdom 和 HTML 的显示窗口大小
                #display_id=0,
                update_html_freq=100,
                save_epoch_freq=5,
                # print_freq=10,
            ),
        ]
    # 定义 commands 方法，返回生成运行实验的命令列表
    def commands(self):
        #return ["python train.py " + str(opt) for opt in self.common_options()]
        return ["python train.py " + str(opt) for opt in self.common_options()]
    # 定义 test_commands 方法，返回生成运行测试的命令列表
    def test_commands(self):
        opts = self.common_options()
        phase = 'val'
        for opt in opts:
            opt.set(crop_size=1024, num_test=1000)
            opt.remove('n_epochs', 'n_epochs_decay', 'update_html_freq',
                       'save_epoch_freq', 'continue_train', 'epoch_count')
        return ["python test.py " + str(opt.set(phase=phase)) for opt in opts]

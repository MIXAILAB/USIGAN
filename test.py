"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util


if __name__ == '__main__':
    opt = TestOptions().parse()  # 解析测试选项
    # 为测试硬编码一些参数
    opt.num_threads = 0   # 测试代码仅支持 num_threads = 1
    opt.batch_size = 1    # 测试代码仅支持 batch_size = 1
    opt.serial_batches = True  # 禁用数据洗牌；如果需要在随机选择的图像上查看结果，请取消注释此行。
    opt.no_flip = True    # 不翻转图像；如果需要在翻转的图像上查看结果，请取消注释此行。
    opt.display_id = -1   # 不使用Visdom显示；测试代码将结果保存到HTML文件中。
    dataset = create_dataset(opt)  # 根据 opt.dataset_mode 和其他选项创建数据集
    train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      # 根据 opt.model 和其他选项创建模型
    # 创建用于查看结果的网页
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # 定义网站目录
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # 常规设置：加载和打印网络；创建调度器
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:  # 仅将我们的模型应用于 opt.num_test 张图像。
            break
        model.set_input(data)  # 从数据加载器中解包数据
        model.test()          # 运行推理
        visuals = model.get_current_visuals()  # 获取图像结果
        img_path = model.get_image_paths()      # 获取图像路径
        if i % 5 == 0:  # 将图像保存到HTML文件
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, width=opt.display_winsize)
    webpage.save()  # 保存HTML

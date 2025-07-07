import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer


if __name__ == '__main__':
    opt = TrainOptions().parse()    # 获取训练选项
    dataset = create_dataset(opt) # 根据opt.dataset_mode和其他选项创建数据集
    dataset_size = len(dataset)   # 获取数据集中的图像数量

    model = create_model(opt)      # 根据opt.model和其他选项创建模型
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # 创建一个用于显示/保存图像和绘制图表的可视化工具
    opt.visualizer = visualizer
    total_iters = 0                # 训练迭代的总数

    optimize_time = 0.1

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # 对不同epoch的外循环；我们通过 <epoch_count>, <epoch_count>+<save_latest_freq> 保存模型
        epoch_start_time = time.time()  # 整个epoch的计时器
        iter_data_time = time.time()    # 每次迭代的数据加载计时器
        epoch_iter = 0                  # 当前epoch中的训练迭代次数，每个epoch都重置为0
        visualizer.reset()              # 重置可视化工具：确保每个epoch至少保存一次结果到HTML

        dataset.set_epoch(epoch)
        for i, data in enumerate(dataset):  # 内循环在一个epoch内
            iter_start_time = time.time()  # 每次迭代的计算计时器
            data['current_epoch'] = epoch
            data['current_iter'] = total_iters
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # 常规设置：加载和打印网络；创建调度程序
                model.parallelize()
            model.set_input(data)  # 从数据集中解包数据并应用预处理
            model.optimize_parameters()   # 计算损失函数，获取梯度，更新网络权重
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            
            optimize_time = (time.time() - optimize_start_time)  / batch_size
        #     break
        # break
            if total_iters % opt.display_freq == 0:   # 在visdom上显示图像并将图像保存到HTML文件
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # 打印训练损失并将日志信息保存到磁盘
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # 每隔 <save_latest_freq> 次缓存我们的最新模型
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # 偶尔在控制台上显示实验名称是有用的
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # 每隔 <save_epoch_freq> 次缓存我们的模型
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # 在每个epoch结束时更新学习率

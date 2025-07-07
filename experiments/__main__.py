import os
import importlib


def find_launcher_using_name(launcher_name):
    # cur_dir = os.path.dirname(os.path.abspath(__file__))
    # pythonfiles = glob.glob(cur_dir + '/**/*.py')
    # 构建launcher模块的文件名
    launcher_filename = "experiments.{}_launcher".format(launcher_name)
    launcherlib = importlib.import_module(launcher_filename)

    # In the file, the class called LauncherNameLauncher() will
    # be instantiated. It has to be a subclass of BaseLauncher,
    # and it is case-insensitive.
    # 在文件中查找名为Launcher的类
    launcher = None
    # target_launcher_name = launcher_name.replace('_', '') + 'launcher'
    for name, cls in launcherlib.__dict__.items():
        if name.lower() == "launcher":
            launcher = cls

    if launcher is None:
        raise ValueError("In %s.py, there should be a class named Launcher")

    return launcher


if __name__ == "__main__":
    import argparse
    # 命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('name')# launcher名称
    parser.add_argument('cmd')# 命令
    parser.add_argument('id', nargs='+', type=str)# ID列表
    parser.add_argument('--mode', default=None)# 模式
    parser.add_argument('--which_epoch', default=None)# 哪个epoch
    parser.add_argument('--continue_train', action='store_true')# 是否继续训练
    parser.add_argument('--subdir', default='')# 子目录
    parser.add_argument('--title', default='')# 标题
    parser.add_argument('--gpu_id', default=None, type=int)# GPU ID
    parser.add_argument('--phase', default='test')# 阶段，默认为测试阶段

    opt = parser.parse_args()

    name = opt.name
    Launcher = find_launcher_using_name(name)

    instance = Launcher()

    cmd = opt.cmd
    ids = 'all' if 'all' in opt.id else [int(i) for i in opt.id]
    if cmd == "launch":
        # 启动实验
        instance.launch(ids, continue_train=opt.continue_train)
    elif cmd == "stop":
        # 停止实验
        instance.stop()
    elif cmd == "send":
        assert False
    elif cmd == "close":
        # 关闭实验
        instance.close()
    elif cmd == "dry":
        # 执行干运行
        instance.dry()
    elif cmd == "relaunch":
        # 重新启动实验
        instance.close()
        instance.launch(ids, continue_train=opt.continue_train)
    elif cmd == "run" or cmd == "train":
        assert len(ids) == 1, '%s is invalid for run command' % (' '.join(opt.id))
        expid = ids[0]
        instance.run_command(instance.commands(), expid,
                             continue_train=opt.continue_train,
                             gpu_id=opt.gpu_id)
    elif cmd == 'launch_test':
        # 启动测试
        instance.launch(ids, test=True)
    elif cmd == "run_test" or cmd == "test":
        # 运行测试
        test_commands = instance.test_commands()
        if ids == "all":
            ids = list(range(len(test_commands)))
        for expid in ids:
            instance.run_command(test_commands, expid, opt.which_epoch,
                                 gpu_id=opt.gpu_id)
            if expid < len(ids) - 1:
                os.system("sleep 5s")
    elif cmd == "print_names":
        # 打印实验名称
        instance.print_names(ids, test=False)
    elif cmd == "print_test_names":
        # 打印测试实验名称
        instance.print_names(ids, test=True)
    elif cmd == "create_comparison_html":
        # 创建对比HTML
        instance.create_comparison_html(name, ids, opt.subdir, opt.title, opt.phase)
    else:
        raise ValueError("Command not recognized")

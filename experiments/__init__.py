import os
import importlib

#这个代码块主要是一个用于动态加载不同launcher的脚本，通过命令行参数选择要加载的launcher，
#然后执行相应的操作（launch、stop、send）
def find_launcher_using_name(launcher_name):
    # cur_dir = os.path.dirname(os.path.abspath(__file__))
    # pythonfiles = glob.glob(cur_dir + '/**/*.py')
    # 构建launcher模块的文件名
    launcher_filename = "experiments.{}_launcher".format(launcher_name)
    # 导入launcher模块
    launcherlib = importlib.import_module(launcher_filename)

    # In the file, the class called LauncherNameLauncher() will
    # be instantiated. It has to be a subclass of BaseLauncher,
    # and it is case-insensitive.
    # 在模块中查找与目标launcher名称匹配的类
    launcher = None
    target_launcher_name = launcher_name.replace('_', '') + 'launcher'
    for name, cls in launcherlib.__dict__.items():
        if name.lower() == target_launcher_name.lower():
            launcher = cls

    if launcher is None:
        raise ValueError("In %s.py, there should be a subclass of BaseLauncher "
                         "with class name that matches %s in lowercase." %
                         (launcher_filename, target_launcher_name))

    return launcher


if __name__ == "__main__":
    import sys
    import pickle

    assert len(sys.argv) >= 3
    # 从命令行参数获取launcher名称
    name = sys.argv[1]
    # 根据名称找到对应的launcher类
    Launcher = find_launcher_using_name(name)
    # 构建launcher实例的缓存路径
    cache = "/tmp/tmux_launcher/{}".format(name)
    # 如果缓存文件存在，则从文件中加载实例，否则创建新的实例
    if os.path.isfile(cache):
        instance = pickle.load(open(cache, 'r'))
    else:
        instance = Launcher()
     # 获取命令
    cmd = sys.argv[2]
    # 根据命令执行相应的操作
    if cmd == "launch":
        instance.launch()
    elif cmd == "stop":
        instance.stop()
    elif cmd == "send":
        # 从命令行参数获取实验ID和命令
        expid = int(sys.argv[3])
        cmd = int(sys.argv[4])
        instance.send_command(expid, cmd)
    # 确保保存实例到缓存文件
    os.makedirs("/tmp/tmux_launcher/", exist_ok=True)
    pickle.dump(instance, open(cache, 'w'))

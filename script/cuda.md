# 1. 安装驱动
```bash
ubuntu-drivers devices # 检查驱动
sudo ubuntu-drivers autoinstall # 自动安装驱动
sudo reboot # 重启系统
nvidia-smi # 查看安装的驱动版本
```
# 2. CUDA工具包
```bash
weget https://developer.nvidia.com/cuda-12-2-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local
sudo sh cuda_12.2.2_535.104.05_linux.run
```
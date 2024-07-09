# conda Error while loading cuda entry point
- 问题原因 conda 24.5.0版本有BUG
- 解决方法 conda install conda=24.3.0 然后 conda clean --all
# conda 创建新环境只有conda-meta
- 问题原因 未知
- 解决方法 创建环境的时候添加python版本号 conda create -n python=3.11
# Jupyter 远程登录

```bash
jupyter notebook --no-browser --port=8080
ssh -L 8080:localhost:8080 jinyalong@113.64.244.23 -p 1204
```
# Jupyter 设置密码
## 利用ipython配置密码
```python
from notebook.auth import passwd
password= passwd()
print(password)
```
##  在 ~/.jupyter/jupyter_notebook_config.py 中找到c.NotebookApp.password 字段将其修改为上一步获得的hash密码：
```python
c.NotebookApp.ip='*'#所有ip可访问
c.NotebookApp.password = u'{password}'#上面的链接
c.NotebookApp.open_browser = False
c.NotebookApp.port =8989
```

# 远程复制
- host: 113.64.244.23:1204
- user：jinyalong qs*@*a$8Rn
```bash
scp -r -P 1204 DNABERT-2-117M jinyalong@113.64.244.23:/data/home/jinyalong/data/
```
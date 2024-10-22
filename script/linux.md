# Jupyter 远程登录

```bash
ps -ef | grep jinya | grep jupyter | awk '{ print $2 }' | xargs kill -9
nohup jupyter notebook --no-browser --port=8080 2>&1 &
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
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
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
```

# 远程复制
- host: 113.64.244.23:1204
- user：jinyalong qs*@*a$8Rn
```bash
scp -r -P 1204 dna/* jinyalong@113.64.244.23:/data/home/jinyalong/notebook/dna
rsync -av -e 'ssh -p 1204' jinyalong@113.64.244.23:/data/home/jinyalong/data/sev_241001/results/core6-merge_core /Users/john/data/Promter/results/core6-merge_core
cat DNA2.csv | grep "AACACGGG***TT" | awk -F, '{sum += $3};END {print sum}'
scp -P 1204 jinyalong@113.64.244.23:/data/home/jinyalong/data/sev_240624/results/fs*.csv ./
```
# 相关论文
- CodeGemma Open Code Models Based on Gemma 
- Gemma Open Models Based on Gemini Research and Technology
# 相关开源项目地址
- https://huggingface.co/google/gemma-2b
- https://huggingface.co/google/codegemma-2b
# huggingface-hub教程
* 安装 登录*
* https://github.com/huggingface/huggingface_hub
```bash
pip install huggingface_hub
huggingface-cli login
# or using an environment variable
huggingface-cli login --token $HUGGINGFACE_TOKEN # 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
huggingface-cli download google/gemma-2b-it-pytorch
huggingface-cli download google/codegemma-2b
```

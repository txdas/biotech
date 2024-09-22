import openai
import os
os.environ["http_proxy"] = "http://127.0.0.1:33210"
os.environ["https_proxy"] = "http://127.0.0.1:33210"
openai_api_key = "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"
os.environ.update({"OPENAI_API_KEY": openai_api_key, "TOKENIZERS_PARALLELISM": "true"})

client = openai.OpenAI()


# 生成代码generate_code
def generate_code(description):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 使用适当的模型
        messages=[
            {
                "role": "system",
                "content": "You will be provided with a define of Python function with its description, "+
                           "your task is to write a python function fulfilling the function description," +
                            "and the python function name is solution_function"
            },
            {
                "role": "user",
                "content": description
            }
        ],
        max_tokens=500,
    )

    response = response.choices[0].message.content.strip()
    return response


def generate_one_completion(description):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 使用适当的模型
        messages=[
            {
                "role": "system",
                "content": "You will be provided with a define of Python function with its description, "+
                           "your task is to complete it. Please don't add other comments or texts"
            },
            {
                "role": "user",
                "content": description
            }
        ],
        max_tokens=500,
    )
    response = response.choices[0].message.content.strip()
    return response
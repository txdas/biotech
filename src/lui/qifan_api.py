import requests
import json

API_KEY = "换成你的API_KEY "
SECRET_KEY = "换成你的SECRET_KEY "


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


def generate(prompt):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token=" + get_access_token()

    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return eval(response.text.replace('false', '"false"'))['result']


def generate_code(description):
    prompt = f'''
    你的任务是根据一段函数的功能描述，写一个实现该功能Python的函数：
    {description} 
    你只需完成函数的实现，不要添加其他注释或说明，同时函数的名称为solution_functio
    '''
    return generate(prompt)


if __name__ == '__main__':
    prompt = '写一个有转折的笑话'
    content = generate(prompt)
    print(content)

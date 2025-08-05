from openai import OpenAI

client = OpenAI(
    api_key="bce-v3/ALTAK-7tdrtNmRWcg4LtcBd8RcD/460f221b8695d9f9b25588a12a306b8887c0a965",  # 千帆bearer token
    base_url="https://qianfan.baidubce.com/v2",  # 千帆域名
    #default_headers={"appid": "app-xxxxxx"}   # 用户在千帆上的appid，非必传
)

completion = client.chat.completions.create(
    model="qianfan-70b", #"meta-llama-3-8b", #"glm-4-32b-0414", #"kimi-k2-instruct", #"deepseek-v3", #"qianfan-70b", #"ernie-4.5-0.3b", #"meta-llama-3-70b", #"ernie-4.0-turbo-8k", # 预置服务请查看模型列表，定制服务请填入API地址
    messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
              {'role': 'user', 'content': 'Hello！'}]
)

print(completion.choices[0].message)
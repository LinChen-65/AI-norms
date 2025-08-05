#%%
import requests
import time
import yaml
from munch import munchify
import pdb
from openai import OpenAI
from cost_tracker import Qwen30BTracker
#%%
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)
# set temperature to 0 for deterministic outcomes
temperature = config.params.temperature
if temperature == 0:
    llm_params = {"do_sample": False,
            "max_new_tokens": 12,
            "return_full_text": False, 
            }
else:
    llm_params = {"do_sample": True,
            "temperature": temperature,
            "top_k": 10,
            "max_new_tokens": 15,
            "return_full_text": False, 
            }  
#%%
API_TOKEN = config.model.API_TOKEN   
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/"+config.model.model_name

# 创建全局 tracker
cost_tracker = Qwen30BTracker()
#%%


def query(client, payload):
    try:
        completion = client.chat.completions.create(
            model=config.model.model_name,
            messages=payload["inputs"],
        )
        #response = completion.choices[0].message
        response = completion
        # === 记录 token 消耗 ===
        usage = response.usage  # API 自动返回
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        cost = cost_tracker.add_usage(input_tokens, output_tokens)
        #print(f"[COST] 本次: ${cost:.6f} | 累计: ${cost_tracker.total_cost_usd:.6f}")
    except:
        return None
    return response, cost_tracker.total_cost_usd
'''
def query(payload):
    "Query the Hugging Face API"
    try:
        response = requests.post(API_URL, headers=headers, json=payload).json()
    except:
        return None
    return response
'''

def get_response(chat, options):
    """Generate a response from the model."""

    client = OpenAI(
        api_key=config.model.API_TOKEN,  # 千帆bearer token
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # Aliyun
        #base_url="https://qianfan.baidubce.com/v2",  # 千帆域名
        #default_headers={"appid": "app-xxxxxx"}   # 用户在千帆上的appid，非必传
    )

    overloaded = 1
    while overloaded == 1:
        response, total_cost = query(client, {"inputs": chat, "parameters": llm_params, "options": {"use_cache": False}})
        #print(response)
        generated_text = response.choices[0].message.content
        if response == None:
            print('CAUGHT JSON ERROR')
            continue

        if type(response)==dict:
            print("AN EXCEPTION: ", response)
            time.sleep(2.5)
            if "Inference Endpoints" in response['error']:
              print("HOURLY RATE LIMIT REACHED")
              time.sleep(450)
                
        #elif any(option in response[0]['generated_text'].split("'") for option in options):.
        elif any(option in generated_text.split("'") for option in options):
            overloaded=0
    #response_split = response[0]['generated_text'].split("'")
    response_split = generated_text.split("'")
    for opt in options:
        try:
            index = response_split.index(opt)
        except:
            continue
    #print(response_split[index])
    return response_split[index], total_cost

def get_meta_response(chat):
    """Generate a response from the Llama model."""

    overloaded = 1
    while overloaded == 1:
        response = query({"inputs": chat, "parameters": llm_params, "options": {"use_cache": False}})
        #print(response)
        if response == None:
            print('CAUGHT JSON ERROR')
            continue

        if type(response)==dict:
            print("AN EXCEPTION")
            time.sleep(2.5)
            if "Inference Endpoints" in response['error']:
              print("HOURLY RATE LIMIT REACHED")
              time.sleep(900)
                
        elif 'value' in response[0]['generated_text']:
            overloaded=0
    
            response_split = response[0]['generated_text'].split(";")
            response_split = response_split[0].split(": ")
            if len(response_split)<2:
                overloaded = 1
    print(response_split[1])
    return response_split[1]
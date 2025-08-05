#from modelscope import snapshot_download

#cache_dir = '/data1/chenlin/'
#model_dir = snapshot_download('LLMs/Meta-Llama-3-70B-Instruct',cache_dir=cache_dir)


import os
import torch
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  

# 递归搜索 config.json
def find_hf_model_dir(base_dir):
    for root, dirs, files in os.walk(base_dir):
        if "config.json" in files:
            cfg_path = os.path.join(root, "config.json")
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if "model_type" in cfg:
                    return root
            except Exception as e:
                pass
    raise FileNotFoundError(f"[ERROR] 在 {base_dir} 下找不到包含 'model_type' 的 config.json，请检查模型文件")


# ======================
# 统一模型保存位置
# ======================
model_name = 'Meta-Llama-3-8B-Instruct' #'Meta-Llama-3-70B-Instruct'
MODEL_ID = f"LLM-Research/{model_name}"
MODEL_CACHE_DIR = f"/data1/chenlin/LLMs/{model_name}"  # 自定义路径


# ======================
# 第一次运行时下载模型
# ======================
if not os.path.exists(MODEL_CACHE_DIR):
    print(f"[INFO] 模型未找到，本地路径: {MODEL_CACHE_DIR}")
    print(f"[INFO] 正在从 ModelScope 下载 {MODEL_ID}...")
    model_dir = snapshot_download(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    print(f"[INFO] 模型已下载到 {model_dir}")
else:
    #model_dir = MODEL_CACHE_DIR
    # 自动找到 HF 格式模型目录
    model_dir = find_hf_model_dir(MODEL_CACHE_DIR)
    #print(f"[INFO] 检测到 HF 模型路径: {model_dir}")
    print(f"[INFO] 直接使用本地缓存模型: {model_dir}")

# ======================
# 配置 4-bit 量化
# ======================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                   # 开启 4-bit 量化
    bnb_4bit_compute_dtype=torch.float16, # 计算类型 FP16
    bnb_4bit_use_double_quant=True,       # 双重量化，进一步节省显存
    bnb_4bit_quant_type="nf4"             # 使用 NF4 量化（效果最好）
)

# ======================
# 加载 tokenizer
# ======================
print("[INFO] 正在加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# ======================
# 加载模型（自动分配到多卡/单卡）
# ======================
print("[INFO] 正在加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,  # 半精度节省显存
    quantization_config=bnb_config,
    device_map="auto"           # 自动选择显卡
)

# ======================
# 测试推理
# ======================
prompt = "请用一句话解释量子计算。"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n=== 模型回答 ===")
print(response)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

# 1. 选择一个小型中文指令模型
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# 2. 检查 GPU
assert torch.cuda.is_available(), "CUDA 不可用"
print("GPU:", torch.cuda.get_device_name(0))

# 3. 加载分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 4. 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
).to("cuda")

print("Base model loaded")

# 5. 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)

# 6. 将 LoRA 适配器添加到模型
model = get_peft_model(model, lora_config)

# 7. 查看可训练参数
model.print_trainable_parameters()

# 8. 检查模型是否能正常前向计算
messages = [
    {"role": "user", "content": "请用一句话解释什么是联邦学习。"}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    prompt,
    return_tensors="pt",
).to("cuda")

model.eval()
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
    )

print("\n模型输出：")
print(tokenizer.decode(output[0], skip_special_tokens=True))

# 9. 保存 LoRA 适配器
model.save_pretrained("outputs/lora_smoke_test")
tokenizer.save_pretrained("outputs/lora_smoke_test")

print("\nLoRA smoke test passed.")
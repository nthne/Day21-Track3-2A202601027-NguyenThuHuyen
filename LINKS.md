# Lab 21 — Artifact Links & Resources

**Học viên**: Nguyễn Thu Huyền  
**MSSV**: 2A202601027  
**Lớp / Module**: AICB-P2T3 · Ngày 21 — Fine-tuning LLMs  

---

## 🔗 Các liên kết chính thức

| Thành phần | Đường dẫn (URL) | Mô tả |
|---|---|---|
| **GitHub Repository** | [https://github.com/nthne/Day21-Track3-2A202601027-NguyenThuHuyen](https://github.com/nthne/Day21-Track3-2A202601027-NguyenThuHuyen) | Toàn bộ mã nguồn, cấu hình, dữ liệu, test suite và báo cáo hoàn chỉnh |
| **Hugging Face Hub Model** | [https://huggingface.co/nthne/lab21-qwen35-triage-vi](https://huggingface.co/nthne/lab21-qwen35-triage-vi) | LoRA Adapter (`correct` run) cho bài toán CSKH Ticket Triage trên Qwen3.5-4B |

---

## 🚀 Hướng dẫn nạp Adapter từ Hugging Face Hub

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "unsloth/Qwen3.5-4B"
ADAPTER_HUB_ID = "nthne/lab21-qwen35-triage-vi"

# 1. Nạp Base Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 2. Nạp LoRA Adapter trực tiếp từ HuggingFace Hub
model = PeftModel.from_pretrained(base_model, ADAPTER_HUB_ID)
model.eval()

# 3. Phân loại ticket kiểm thử
ticket = "Shop ơi, mình đặt bàn phím cơ mã đơn DH123456. Giao hàng chậm. Đã 3 ngày rồi. Nhờ shop kiểm tra."
prompt = f"<|im_start|>user\n{ticket}<|im_end|>\n<|im_start|>assistant\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=160, do_sample=False)

print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
# Kết quả: {"intent": "van_chuyen", "urgency": "trung_binh", "product": "bàn phím cơ", "sentiment": "trung_tinh"}
```

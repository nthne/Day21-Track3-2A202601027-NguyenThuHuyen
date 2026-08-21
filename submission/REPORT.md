# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Thu Huyền  **MSSV**: 2A202601027  **Ngày**: 2026-08-21  
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (14.6 GB khả dụng)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 steps |

**Template có giữ khối `<think>` không?** `có` — *(results/template_check.json: open_tag_present=True, body_present=True)*.  
Chat template của Qwen3.5 bảo toàn khối `<think>` khi huấn luyện trên dữ liệu suy luận (reasoning traces). Tuy nhiên, trên tập dữ liệu CSKH 250 mẫu chuẩn (bare JSON), chat template tự động đóng cặp thẻ rỗng `<think>\n\n</think>\n\n` ngay trong generation prompt, do đó vùng supervised span chỉ chứa thuần túy chuỗi JSON kết quả của assistant.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 (41.49%) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0000 | 0.7578 | 0.0000 | 3197.9 |
| (b) base + optimized prompt | 0.7650 | 0.7578 | 1.0000 | 1001.9 |
| (c) LoRA fine-tune | 0.9600 | 0.6778 | 1.0000 | 1423.8 |

**(b) có thật sự mạnh hơn (a) không?** `có` — target tăng từ 0.0000 lên 0.7650 và tỷ lệ đúng định dạng format đạt tuyệt đối 1.0000 (100%).  
Không sửa `OPTIMIZED_PROMPT` nhằm giữ nguyên tính khách quan của phép đo baseline đóng băng (`optimized_prompt_sha: 719e74d3b6232053`). Prompt tối ưu (b) có định nghĩa schema chi tiết kèm ví dụ minh họa cụ thể, giúp base model sinh đúng cấu trúc JSON 4 trường vượt trội hoàn toàn so với prompt sơ sài (a).

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 (1e-4) | 0.6276 | 0.9600 | 923.1 | 12.01 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 0.0001 (1e-4) | 0.5360 | 0.9700 | 778.2 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 (1e-5) | 1.5704 | 0.0000 | 911.9 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 (1e-4) | 0.7058 | 0.9400 | 1002.1 | 7.09 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**

Run `attn_only` được nâng rank lên $r=283$ để khớp chính xác ngân sách tham số huấn luyện (~32.46M params) với `correct` ($r=16$). Trên tập đánh giá `target`, `attn_only` đạt 0.9700 (thắng nhẹ `correct` ở mức 0.9600) và có train loss thấp hơn (0.5360 so với 0.6276). Tuy nhiên, việc phải dồn rank lên tới 283 ở các lớp attention để bù đắp cho việc thiếu vắng adapter ở các lớp MLP/linear khác cho thấy dung lượng tham số bị phân bổ lệch; đòn bẩy thực sự là tổng ngân sách tham số và năng lực biểu diễn của dữ liệu, chứ không phải bản thân con số rank cao.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

Run `wrong_lr` sử dụng learning rate $1\times 10^{-5}$ (thang đo của Full Fine-Tuning thay vì $1\times 10^{-4}$ chuẩn cho LoRA), khiến đường loss gần như phẳng lì và dừng lại ở mức rất cao 1.5704. Kết quả là mô hình đạt điểm 0.0000 trên tập target và tỷ lệ format là 0.0000 (hoàn toàn không học được cú pháp JSON). Nếu chỉ nhìn vào loss mà không biết nguyên nhân do LR quá nhỏ, người làm sẽ dễ kết luận sai lầm rằng cấu hình LoRA không đủ khả năng học bài toán hoặc tập dữ liệu bị gán nhãn sai.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**

Cấu hình `qlora` (4-bit NF4) tiết kiệm được 4.92 GB VRAM đỉnh (giảm từ 12.01 GB xuống 7.09 GB, tương đương giảm ~41% VRAM). Tuy nhiên, cái giá phải trả là thời gian huấn luyện tăng lên 1002.1s (chậm hơn 923.1s do chi phí dequantization on-the-fly) và độ chính xác target bị sụt giảm từ 0.9600 xuống 0.9400. Kết quả đo lường này hoàn toàn củng cố khuyến cáo từ nhà phát triển mô hình: trên kiến trúc Qwen3.5, sai số lượng tử hóa 4-bit làm suy hao chất lượng đầu ra, do đó nên sử dụng 16-bit LoRA (fp16/bf16) khi phần cứng đáp ứng đủ VRAM.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.1950` · `regression Δ = -0.0800` · `valid_trace_rate = 0.00`

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?  
Cổng hồi quy đưa ra phán quyết `FAILED` do điểm số năng lực tổng quát (`regression`) bị sụt giảm $-0.0800$ (từ 0.7578 xuống 0.6778), vượt quá ngưỡng dung sai cho phép là 0.0200 (`REGRESSION_TOLERANCE`), dù độ chính xác nhiệm vụ chính (`target`) tăng trưởng mạnh mẽ $+0.1950$ (+19.5% so với baseline prompt tối ưu). 

Hiện tượng này phản ánh trực tiếp quy luật **Quên Thảm Họa (Catastrophic Forgetting)** và **Thuế Căn Chỉnh (Alignment Tax)** được phân tích trong Deck §14.3: khi fine-tune một mô hình ngôn ngữ lớn trên một tập dữ liệu hẹp (250 ticket CSKH đơn miền) mà không đưa dữ liệu hồi tưởng (replay data), các trọng số LoRA bị tối ưu hóa quá mức vào cấu trúc JSON chuyên biệt, dẫn đến suy giảm khả năng trả lời các câu hỏi tri thức phổ thông. Để khắc phục triệt để và đưa Cổng Hồi Quy về trạng thái `PASSED`, giải pháp bắt buộc theo lý thuyết là phải trộn thêm **1–5% dữ liệu đàm thoại/tri thức tổng quát** vào tập huấn luyện (general replay buffer) nhằm bảo toàn năng lực nền tảng của base model.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt đèn bàn LED mã đơn VN339109. Vỡ khi nhận. Gấp. | `san_pham_loi`, `cao`, `đèn bàn LED`, `trung_tinh` | Format chuẩn nhưng intent phân vân | `{"intent": "san_pham_loi", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "trung_tinh"}` | ✅ FT thắng: Nhận diện chính xác sản phẩm lỗi hỏng và gán đúng độ khẩn cấp cao. |
| 2 | Xin chào, mình đặt balo laptop mã đơn DH863123. Đổi size. Hỏi cho biết | `doi_tra`, `thap`, `balo laptop`, `tieu_cuc` | Trả lời dài dòng, format dư thừa | `{"intent": "doi_tra", "urgency": "thap", "product": "balo laptop", "sentiment": "tieu_cuc"}` | ✅ FT thắng: Trích xuất đúng ý định đổi trả và nhận diện sentiment tiêu cực chuẩn. |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. | `hoan_tien`, `trung_binh`, `bình giữ nhiệt`, `tieu_cuc` | Trích xuất đủ 4 trường JSON | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment":` | ❌ **FT thua**: Chuỗi JSON bị cắt cụt ở trường sentiment cuối cùng (ft_score=0.75). |
| 4 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. | `san_pham_loi`, `trung_binh`, `nồi chiên không dầu`, `tieu_cuc` | Nhận diện đúng intent và format | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sen` | ❌ **FT thua**: Bị ngắt token giữa chừng ở trường sentiment do chạm ngưỡng max tokens (ft_score=0.75). |
| 5 | Alo shop, mình đặt máy xay sinh tố mã đơn VN724342. Trả lại tiền. | `hoan_tien`, `trung_binh`, `máy xay sinh tố`, `tieu_cuc` | Trả về JSON hoàn chỉnh | `{"intent": "doi_tra", "urgency": "thap", "product": "máy xay sinh tố", "sentiment": "trung` | ❌ **FT thua**: Phân loại nhầm sang `doi_tra` thay vì `hoan_tien` và bị cắt cụt sentiment (ft_score=0.75). |

**Có mẫu chung nào ở các ca FT thua không?**  
Mẫu chung ở hầu hết các ca FT thua (đạt 0.75 thay vì 1.0) là mô hình nhận diện chính xác 2–3 trường đầu tiên (`product`, `urgency`, `intent`), nhưng chuỗi sinh ra bị nghẽn token ở trường `sentiment` cuối cùng hoặc bị nhầm lẫn giữa hai intent tương đồng (`doi_tra` vs `hoan_tien`).

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).**  
Bản fine-tune LoRA (`correct`) chứng minh năng lực vượt trội trên bài toán phân loại ticket CSKH khi nâng độ chính xác target lên **96.00%** (vượt baseline prompt tối ưu 19.5% và vượt base model thô 96%) với tỷ lệ tuân thủ cú pháp JSON đạt 100%. Tuy nhiên, kết quả Cổng Hồi Quy `FAILED` ($\Delta \text{regression} = -0.0800$) đưa ra một cảnh báo kỹ thuật then chốt: mô hình không nên được triển khai thẳng vào production như một chatbot đa năng độc lập, mà cần được đóng gói như một **microservice chuyên trách (Task-Specific Adapter)** hoặc phải được tái huấn luyện kèm **1–5% dữ liệu hồi tưởng (replay data)** để bù đắp năng lực ngôn ngữ tổng quát.

Lab đã làm sáng tỏ bản chất của các đòn bẩy kỹ thuật:
1. **Loss Mask chính xác là điều kiện tiên quyết:** Phải chứng minh được loss chỉ tính trên nhãn đầu ra bằng giải mã ngược; rò rỉ prompt vào loss sẽ phá hủy hoàn toàn mô hình.
2. **Thang Learning Rate quyết định sự hội tụ:** LoRA bắt buộc phải dùng LR thang $10^{-4}$ ($\approx 10\times$ full-FT) để các ma trận adapter cập nhật hiệu quả.
3. **Phân bổ adapter toàn diện (`all-linear`):** Đặt adapter trên toàn bộ text decoder giúp phân tán biểu diễn tri thức ổn định hơn việc ép rank cực đại vào riêng attention.

**Ba điều tôi học được** (cụ thể, không generic):
1. **Tính liêm chính của phép so sánh (Fair Benchmark):** Baseline prompt (b) phải được thiết kế tối ưu và đo đạc *trước* khi train; so sánh các biến thể kiến trúc phải cố định ngân sách tham số (`matched_rank`) và số optimizer steps.
2. **Chấm điểm bằng metric nhiệm vụ thay vì train loss:** Train loss thấp hơn (như ở `attn_only` loss 0.5360) không đồng nghĩa với mô hình tổng quát hóa tốt hơn; metric trên tập eval độc lập mới là thước đo chân thực.
3. **Cổng hồi quy (Regression Gate) là chốt chặn an toàn:** Đánh giá LLM sau fine-tune bắt buộc phải kiểm tra hiện tượng suy thoái tri thức nền tảng (catastrophic forgetting) để đưa ra quyết định triển khai có trách nhiệm.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
1. Trộn thêm 3% mẫu dữ liệu đa năng từ tập alpaca/instruction vào tập train để chứng minh Cổng Hồi Quy chuyển từ `FAILED` sang `PASSED` mà không làm giảm điểm target 96%.
2. Thực hiện Thử thách B1: Merge adapter vào base model, kiểm tra độ suy hao điểm sau merge và đo đạc độ trễ serving khi hot-swap nhiều adapter trên cùng một instance.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:

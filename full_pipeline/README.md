# GenMask-SAM3: full pipeline

Tinh chỉnh có mục tiêu cho phân đoạn ngữ nghĩa của SAM3, gồm 3 bước:

1. Chọn các lớp yếu (lớp mục tiêu).
2. Huấn luyện một mạng refine nhẹ trên mask thô của SAM3 + ảnh RGB (+ đặc trưng DINOv2).
3. Hợp nhất kết quả mạng refine với dự đoán SAM3 gốc ("hybrid").

Mỗi dataset nằm trong một thư mục riêng, chạy độc lập:

| Thư mục | Dataset | Trạng thái |
|---|---|---|
| `pc59/` | PASCAL-Context 59 (VOC2010) | **Dùng ngay.** Hướng dẫn bên dưới |
| `voc/` | Pascal VOC 2012 | Đã có code, chưa chạy thử trên server |
| `cityscapes/` | Cityscapes | Đã có code, chưa chạy thử trên server |

Tài liệu này chỉ hướng dẫn **PC59, nhánh LLM**.

---

## 1. Pipeline làm gì

Lệnh `run_pipeline_pc59.py` chạy lần lượt 6 bước cho mỗi nhánh prompt (`nollm`, `llm`):

| # | Script | Việc làm | Output (trong `outputs/<arm>/`) |
|---|---|---|---|
| 1 | `sam3_baseline_pc59.py` | Chạy SAM3 trên tập val **một lần**: tính baseline và lưu cache cho bước val | `baseline/`, `val_cache/` |
| 2 | `get_coarse_pc59.py` | Tạo mask thô SAM3 cho các lớp mục tiêu trên tập train. Tạo `outputs/split.json` nếu chưa có | `coarse_cache/` |
| 3 | `train_unetaspp_pc59.py` | Huấn luyện UNet+ASPP | `unetaspp/checkpoint/` |
| 4 | `train_unetasppdinov2_pc59.py` | Huấn luyện UNet+ASPP+DINOv2 (cache DINOv2 dùng chung cho 2 nhánh) | `unetasppdinov2/checkpoint/` |
| 5 | `val_unetaspp_pc59.py` | Đánh giá hybrid: đọc `val_cache`, **không chạy lại SAM3** | `unetaspp/val/` |
| 6 | `val_unetasppdinov2_pc59.py` | Như bước 5, cho kiến trúc DINOv2 | `unetasppdinov2/val/` |

Cuối cùng pipeline ghi `outputs/ablation_table.csv`.

- **Nhánh `nollm`**: prompt là tên lớp.
- **Nhánh `llm`**: prompt lấy từ `configs/adjust_prompt_pc59.json`, chỉ áp dụng cho các lớp mục tiêu. Các lớp khác vẫn dùng tên lớp, giống nhánh `nollm`.

### Chống lỗi "nhánh LLM nhưng prompt vẫn là tên lớp"

- Danh sách lớp mục tiêu **chỉ** lấy từ `configs/target_classes.json`. Không bước nào tự ghi file prompt.
- Trước khi chạy, nhánh LLM **dừng hẳn** (không chỉ cảnh báo) nếu `adjust_prompt_pc59.json`:
  - không tồn tại;
  - có tập lớp khác `target_classes.json`;
  - có lớp mang danh sách prompt rỗng;
  - là file giữ chỗ, tức **mọi** lớp đều chỉ có prompt đúng bằng tên lớp.
- Mỗi bước in đường dẫn tuyệt đối của file prompt kèm bảng `lớp → prompts`. Bảng này cũng được ghi vào `manifest.json` của bước và vào checkpoint.
- Mỗi output có một **fingerprint** (lớp mục tiêu + prompt + cấu hình). Đổi prompt thì cache/checkpoint cũ tự bị coi là chưa xong và được chuyển sang `*.stale-<thời gian>`, không bị dùng lại.

---

## 2. Cấu trúc `pc59/`

```
pc59/
├── run_pipeline_pc59.py          <- FILE CHẠY
├── pc59_common.py                # đặc thù PC59 (mục A) + tiện ích chung (mục B)
├── pc59_refine.py                # model, DINOv2, dataset, loss, vòng train, val hybrid
├── sam3_baseline_pc59.py         # bước 1
├── get_coarse_pc59.py            # bước 2
├── train_unetaspp_pc59.py        # bước 3
├── train_unetasppdinov2_pc59.py  # bước 4
├── val_unetaspp_pc59.py          # bước 5
├── val_unetasppdinov2_pc59.py    # bước 6
├── requirements.txt
├── configs/                      # đã có sẵn trong repo
│   ├── target_classes.json       # 34 lớp (select_refinement_classes.py: GMM, ΔBIC = 6.34)
│   ├── contexts_class_pc59.json  # ngữ cảnh viết tay cho LLM
│   └── adjust_prompt_pc59.json   # prompt LLM THẬT (vd. shelves -> shelves, shelf, bookshelf, rack)
├── tools/                        # chạy tay, không nằm trong pipeline
│   ├── check_pc59_class_order.py
│   ├── select_refinement_classes.py
│   ├── generate_adjust_prompt_pc59.py
│   └── prepare_pc59_mat_to_png.ipynb
├── data/                         # KHÔNG có trong git, tự đặt vào (mục 3)
├── weights/                      # KHÔNG có trong git: sam3.pt
└── outputs/                      # pipeline tự sinh
```

Thư mục `voc/` và `cityscapes/` có cùng khuôn. Ngoài mục A của `<ds>_common.py`, các file giống hệt nhau, chỉ khác tên dataset.

---

## 3. Chuẩn bị trên server

### 3.1 Code

```bash
git clone <repo> && cd <repo>/full_pipeline/pc59
```

### 3.2 Dữ liệu (`pc59/data/`)

```
data/
├── JPEGImages/                 # ảnh VOC2010: <id>.jpg
├── SegmentationClassContext/   # mask PNG: 0 = nền, 1..59 = lớp theo alphabet (tools/prepare_pc59_mat_to_png.ipynb)
├── pascal_context_train.txt    # 4998 dòng "JPEGImages/<id>.jpg GroundTruth_trainval_png/<id>.png"
└── pascal_context_val.txt      # 5105 dòng
```

Nếu dữ liệu đã nằm ở chỗ khác trên server, có thể dùng symlink thay vì copy:

```bash
ln -s /datastore/.../JPEGImages data/JPEGImages
```

`59_labels.txt` **không cần**. Thứ tự 59 lớp được viết cứng theo alphabet trong `pc59_common.py`.

### 3.3 Checkpoint SAM3

Đặt file vào `weights/sam3.pt`. Có thể symlink, hoặc chỉ ra đường dẫn bằng `--sam3-ckpt /path/sam3.pt` hay biến môi trường `SAM3_CKPT=/path/sam3.pt`.

DINOv2 (ViT-S/14) và trọng số ImageNet của ResNet-34 tự tải về qua mạng ở lần chạy đầu.

### 3.4 (Nên làm) Dùng lại kết quả của nhánh no-LLM đã chạy

Không bắt buộc, nhưng giúp tiết kiệm thời gian và bảo đảm so sánh công bằng giữa 2 nhánh.

```bash
mkdir -p outputs/baseline_nollm
# 1) cùng split train/val nội bộ với nhánh no-LLM (pipeline kiểm tra file khớp với danh sách train)
cp <thư mục cũ>/coarse_cache_pc59_nollm/split.json outputs/split.json
# 2) cache DINOv2 (chỉ phụ thuộc ảnh) -> bỏ qua ~5000 lần forward DINOv2
cp -r <thư mục cũ>/dinov2_cache_pc59_nollm outputs/dinov2_cache
# 3) kết quả baseline no-LLM cũ, để ablation_table.csv có dòng so sánh
cp <baseline cũ>/per_class_metrics.csv <baseline cũ>/summary_metrics.csv outputs/baseline_nollm/
```

**Không** copy các cache hay checkpoint `*_pc59_llm*` cũ (`coarse_cache_pc59`, `weights_*_llm_v1`, `dinov2_cache_pc59`): chúng được build từ prompt giữ chỗ, tức là sai.

### 3.5 Môi trường Python

```bash
python -m venv venv && source venv/bin/activate
pip install torch torchvision                                   # đúng bản CUDA của server
pip install 'git+https://github.com/facebookresearch/sam3.git' --no-deps
pip install -r requirements.txt
```

---

## 4. Chạy PC59, nhánh LLM

Chạy từ bên trong `full_pipeline/pc59/`.

**Bước 1. Kiểm tra (không tốn GPU, vài giây).** Lệnh này kiểm tra dữ liệu, checkpoint, thư viện và file prompt, rồi in bảng prompt và kế hoạch chạy:

```bash
python run_pipeline_pc59.py --skip-nollm --dry-run
```

Trong output phải thấy prompt thật, ví dụ `shelves -> ['shelves', 'shelf', 'bookshelf', 'rack']`, và dòng `Preflight OK`.

**Bước 2. Smoke test (vài phút).** Chạy trọn 6 bước trên 5 ảnh train, 5 ảnh val, 1 epoch. Kết quả ghi vào `outputs_smoke/`, không đụng tới `outputs/`:

```bash
python run_pipeline_pc59.py --skip-nollm --limit 5
```

**Bước 3. Chạy thật.** Nên chạy trong `tmux` hoặc `nohup`:

```bash
tmux new -s pc59llm
python run_pipeline_pc59.py --skip-nollm
```

Nếu bị ngắt giữa chừng, **chạy lại đúng lệnh cũ**:
- bước nào xong rồi sẽ `skip`;
- `sam3_baseline` và `get_coarse` chạy tiếp từ ảnh đang dở;
- `train` bị ngắt sẽ train lại kiến trúc đó từ đầu, giống bản cũ.

Log đầy đủ nằm trong `outputs/logs/run_<thời gian>.log`.

### Các tùy chọn

| Cờ | Tác dụng |
|---|---|
| `--skip-nollm` / `--skip-llm` | bỏ một nhánh |
| `--archs unetaspp` | chỉ chạy một kiến trúc (mặc định cả hai) |
| `--force train val` | ép chạy lại các bước này (output cũ bị chuyển sang `*.stale-*`). Chọn trong `baseline coarse train val` |
| `--dry-run` | chỉ kiểm tra và in kế hoạch |
| `--limit N` | smoke test trên N ảnh, ghi vào `outputs_smoke/` |
| `--sam3-ckpt PATH` | checkpoint SAM3 nằm ngoài `weights/` |
| `--table-only` | chỉ tạo lại `outputs/ablation_table.csv` |

Từng bước cũng chạy riêng được, ví dụ:

```bash
python sam3_baseline_pc59.py --arm llm
python val_unetasppdinov2_pc59.py --arm llm
```

---

## 5. Kết quả

```
outputs/
├── split.json                    # split train/val nội bộ (dùng chung 2 nhánh)
├── dinov2_cache/                 # dùng chung 2 nhánh
├── baseline_nollm/               # (tự copy vào) baseline no-LLM cũ
├── llm/
│   ├── baseline/                 # SAM3 baseline với prompt LLM
│   ├── val_cache/                # dự đoán SAM3 + mask thô trên val (dùng cho bước 5, 6)
│   ├── coarse_cache/             # mask thô trên train
│   ├── class_pixel_stats.json, sample_coarse_masks.png
│   ├── unetaspp/
│   │   ├── checkpoint/           # best.pth, last.pth, history.json, training_curves.png
│   │   └── val/                  # kết quả hybrid
│   └── unetasppdinov2/           # tương tự
├── ablation_table.csv
└── logs/
```

Mỗi thư mục `baseline/` và `val/` gồm:

- `summary_metrics.csv`: Pixel Accuracy, mIoU (59 lớp), Mean Dice, **mIoU_w** (34 lớp mục tiêu), **mIoU_s** (25 lớp còn lại).
- `per_class_metrics.csv`: IoU, Dice, GT Pixels, Pred Pixels của từng lớp. File val có thêm cột `Source` cho biết lớp đó lấy từ `SAM3 + <kiến trúc> (hybrid)` hay `SAM3 (baseline)`.
- `per_class_iou_bar_chart.png`.
- `class_visualizations/`: 2 ảnh mẫu cho mỗi lớp.
- `manifest.json`: fingerprint, prompt đã dùng, số ảnh.

`ablation_table.csv` có mỗi dòng là một cấu hình (baseline hoặc hybrid × nhánh × kiến trúc), với các cột PA, mIoU, mDice, mIoU_w, mIoU_s.

### Quy ước tính metric (giữ nguyên như bản đã chạy)

- PC59 dùng nhãn 0..58 cho lớp và 255 cho ignore. Pixel nền (raw 0) bị bỏ qua.
- Pixel mà SAM3 không gán cho lớp nào mang giá trị 255 và **không** được tính vào confusion matrix.
- mIoU là trung bình trên đủ 59 lớp.
- Ngưỡng SAM3: một mask tham gia cạnh tranh giữa các lớp khi score > 0.30. Mask thô dùng union-at-threshold với các ngưỡng [0.5, 0.3, 0.2, 0.15].

---

## 6. Kiểm tra nhanh

```bash
python tools/check_pc59_class_order.py      # 59 lớp alphabet, offset raw/canonical, vài mask GT thật
```

Để so với checkpoint no-LLM cũ, chạy val hybrid mới trên checkpoint cũ (cần `outputs/nollm/val_cache` từ `sam3_baseline_pc59.py --arm nollm`):

```bash
python val_unetaspp_pc59.py --arm nollm --ckpt /path/weights_unetaspp_pc59_nollm_v1/unetaspp_pc59_nollm_v1_best.pth
# -> outputs/nollm/unetaspp/val_external/, số liệu phải khớp với kết quả val no-LLM cũ
```

---

## 7. Lỗi thường gặp

| Thông báo | Cách xử lý |
|---|---|
| `LLM arm: ... is a PLACEHOLDER` | `configs/adjust_prompt_pc59.json` là file giữ chỗ. Thay bằng file thật |
| `classes in adjust_prompt_pc59.json do not match target_classes.json` | Hai file lệch nhau. Sửa cho khớp, hoặc sinh lại prompt bằng `tools/generate_adjust_prompt_pc59.py` |
| `split.json does not match the train list` | `outputs/split.json` không phải của PC59. Xóa đi để pipeline tạo lại (seed 42) |
| `[stale] ... moved ... .stale-<time>` | Output cũ được build với cấu hình khác nên đã được chuyển sang tên `*.stale-*`. Xóa khi không cần nữa |
| `DINOv2 unavailable` | Server không có mạng. Đặt repo và trọng số DINOv2 vào `weights/dinov2-repo/` và `weights/dinov2_vits14_pretrain.pth` |

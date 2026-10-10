# GenMask-SAM3: full pipeline

Tinh chỉnh có mục tiêu cho phân đoạn ngữ nghĩa của SAM3, gồm 3 bước:

1. Chọn các lớp yếu (lớp mục tiêu).
2. Huấn luyện một mạng refine nhẹ trên mask thô của SAM3 + ảnh RGB (+ đặc trưng DINOv2).
3. Hợp nhất kết quả mạng refine với dự đoán SAM3 gốc ("hybrid").

Mỗi dataset nằm trong một thư mục riêng, tự chứa đủ mọi thứ và chạy độc lập:

| Thư mục | Dataset | Lớp mục tiêu | Lệnh chạy | Hướng dẫn |
|---|---|---|---|---|
| `pc59/` | PASCAL-Context 59 (ảnh VOC2010) | 34/59 | `python run_pipeline_pc59.py` | [mục 5](#5-pc59--pascal-context-59) |
| `voc/` | Pascal VOC 2012 | 6/20 | `python run_pipeline_voc.py` | [mục 6](#6-voc--pascal-voc-2012) |
| `cityscapes/` | Cityscapes (19 lớp) | 14/19 | `python run_pipeline_cityscapes.py` | [mục 7](#7-cityscapes) |

Thứ tự chạy dự kiến: chạy **PC59, nhánh LLM** trước. Sau khi có kết quả thì chạy **VOC** và **Cityscapes** (cả 2 nhánh).

---

## 1. Pipeline làm gì

Mỗi dataset có một lệnh `run_pipeline_<ds>.py`. Lệnh này chạy lần lượt 6 bước cho mỗi nhánh prompt (`nollm`, `llm`):

| # | Script | Việc làm | Output (trong `outputs/<arm>/`) |
|---|---|---|---|
| 1 | `sam3_baseline_<ds>.py` | Chạy SAM3 trên tập val **một lần**: tính baseline và lưu cache cho bước val | `baseline/`, `val_cache/` |
| 2 | `get_coarse_<ds>.py` | Tạo mask thô SAM3 cho các lớp mục tiêu trên tập train. Tạo `outputs/split.json` nếu chưa có | `coarse_cache/` |
| 3 | `train_unetaspp_<ds>.py` | Huấn luyện UNet+ASPP | `unetaspp/checkpoint/` |
| 4 | `train_unetasppdinov2_<ds>.py` | Huấn luyện UNet+ASPP+DINOv2 (cache DINOv2 dùng chung cho 2 nhánh) | `unetasppdinov2/checkpoint/` |
| 5 | `val_unetaspp_<ds>.py` | Đánh giá hybrid: đọc `val_cache`, **không chạy lại SAM3** | `unetaspp/val/` |
| 6 | `val_unetasppdinov2_<ds>.py` | Như bước 5, cho kiến trúc DINOv2 | `unetasppdinov2/val/` |

Cuối cùng pipeline ghi `outputs/ablation_table.csv`.

- **Nhánh `nollm`**: prompt là tên lớp.
- **Nhánh `llm`**: prompt lấy từ `configs/adjust_prompt_<ds>.json`, chỉ áp dụng cho các lớp mục tiêu. Các lớp khác vẫn dùng tên lớp, giống nhánh `nollm`.
- Ở nhánh `nollm`, bước 1 vẫn chạy vì bước val cần `val_cache`. Baseline no-LLM tính ra ở bước này phải khớp với kết quả đã có, nên cũng dùng được để kiểm tra lại.

### Chống lỗi "nhánh LLM nhưng prompt vẫn là tên lớp"

- Danh sách lớp mục tiêu **chỉ** lấy từ `configs/target_classes.json`. Không bước nào tự ghi file prompt.
- Trước khi chạy, nhánh LLM **dừng hẳn** (không chỉ cảnh báo) nếu `adjust_prompt_<ds>.json`:
  - không tồn tại;
  - có tập lớp khác `target_classes.json`;
  - có lớp mang danh sách prompt rỗng;
  - là file giữ chỗ, tức **mọi** lớp đều chỉ có prompt đúng bằng tên lớp.
- Mỗi bước in đường dẫn tuyệt đối của file prompt kèm bảng `lớp → prompts`. Bảng này cũng được ghi vào `manifest.json` của bước và vào checkpoint.
- Mỗi output có một **fingerprint** (lớp mục tiêu + prompt + cấu hình). Đổi prompt thì cache/checkpoint cũ tự bị coi là chưa xong và được chuyển sang `*.stale-<thời gian>`, không bị dùng lại.

---

## 2. Cấu trúc chung của một thư mục dataset

Ba thư mục có cùng khuôn. Ví dụ với `<ds>` = `pc59`, `voc` hoặc `cityscapes`:

```
<ds>/
├── run_pipeline_<ds>.py          <- FILE CHẠY
├── <ds>_common.py                # mục A: đặc thù dataset (đường dẫn, lớp, nhãn, hyperparameter)
│                                 # mục B: tiện ích chung, giống hệt giữa 3 thư mục
├── <ds>_refine.py                # model, DINOv2, dataset, loss, vòng train, val hybrid
├── sam3_baseline_<ds>.py         # bước 1
├── get_coarse_<ds>.py            # bước 2
├── train_unetaspp_<ds>.py        # bước 3
├── train_unetasppdinov2_<ds>.py  # bước 4
├── val_unetaspp_<ds>.py          # bước 5
├── val_unetasppdinov2_<ds>.py    # bước 6
├── requirements.txt
├── configs/                      # ĐÃ CÓ trong repo
│   ├── target_classes.json
│   ├── contexts_class_<ds>.json
│   └── adjust_prompt_<ds>.json
├── tools/                        # chạy tay, không nằm trong pipeline
├── data/                         # KHÔNG có trong git: tự đặt dữ liệu vào (xem mục của từng dataset)
├── weights/                      # KHÔNG có trong git: sam3.pt
└── outputs/                      # pipeline tự sinh
```

Ngoài mục A của `<ds>_common.py`, các file giống hệt nhau giữa 3 thư mục, chỉ khác tên dataset.

---

## 3. Cài đặt (chung cho cả 3 dataset)

### 3.1 Code

```bash
git clone <repo> && cd <repo>/full_pipeline
```

### 3.2 Môi trường Python

Một môi trường dùng chung cho cả 3 dataset:

```bash
python -m venv venv && source venv/bin/activate
pip install torch torchvision                                   # đúng bản CUDA của server
pip install 'git+https://github.com/facebookresearch/sam3.git' --no-deps
pip install -r pc59/requirements.txt                            # 3 file requirements giống nhau
```

### 3.3 Checkpoint SAM3

Tải tại https://drive.google.com/file/d/1FiUmJKX-CFvkKdcKecsdwikGJ2kVO7Eu/view?usp=sharing

Mỗi dataset tìm checkpoint ở `<ds>/weights/sam3.pt`. Để khỏi phải giữ 3 bản copy của cùng một file, chọn **một** trong các cách sau:

```bash
# cách 1: đặt 1 file, tạo symlink cho từng dataset
mkdir -p pc59/weights voc/weights cityscapes/weights
cp /path/to/sam3.pt pc59/weights/sam3.pt
ln -s "$(pwd)/pc59/weights/sam3.pt" voc/weights/sam3.pt
ln -s "$(pwd)/pc59/weights/sam3.pt" cityscapes/weights/sam3.pt

# cách 2: biến môi trường (có hiệu lực cho mọi dataset)
export SAM3_CKPT=/path/to/sam3.pt

# cách 3: truyền trực tiếp vào lệnh chạy
python run_pipeline_voc.py --sam3-ckpt /path/to/sam3.pt
```

DINOv2 (ViT-S/14) và trọng số ImageNet của ResNet-34 tự tải về qua mạng ở lần chạy đầu.

---

## 4. Quy trình chạy (giống nhau cho mọi dataset)

Luôn chạy từ **bên trong** thư mục dataset (`cd pc59`, `cd voc` hoặc `cd cityscapes`). Làm theo 3 bước:

```bash
# 1) Kiểm tra: không tốn GPU, vài giây. Kiểm tra dữ liệu, checkpoint, thư viện, prompt; in kế hoạch chạy
python run_pipeline_<ds>.py [--skip-nollm | --skip-llm] --dry-run

# 2) Smoke test: vài phút. Chạy trọn các bước trên 5 ảnh train, 5 ảnh val, 1 epoch -> outputs_smoke/
python run_pipeline_<ds>.py [--skip-nollm | --skip-llm] --limit 5

# 3) Chạy thật
python run_pipeline_<ds>.py [--skip-nollm | --skip-llm]
```

- Bước 1 phải in dòng `Preflight OK`. Nếu có chạy nhánh LLM, bảng prompt in ra phải là **prompt thật**.
- Nếu bị ngắt giữa chừng, **chạy lại đúng lệnh cũ**:
  - bước nào xong rồi sẽ `skip`;
  - `sam3_baseline` và `get_coarse` chạy tiếp từ ảnh đang dở;
  - `train` bị ngắt sẽ train lại kiến trúc đó từ đầu.
- Log đầy đủ nằm trong `outputs/logs/run_<thời gian>.log`.

### Các tùy chọn

| Cờ | Tác dụng |
|---|---|
| `--skip-nollm` / `--skip-llm` | bỏ một nhánh (mặc định chạy cả hai) |
| `--archs unetaspp` | chỉ chạy một kiến trúc (mặc định cả hai) |
| `--force train val` | ép chạy lại các bước này (output cũ bị chuyển sang `*.stale-*`). Chọn trong `baseline coarse train val` |
| `--dry-run` | chỉ kiểm tra và in kế hoạch |
| `--limit N` | smoke test trên N ảnh, ghi vào `outputs_smoke/` |
| `--sam3-ckpt PATH` | checkpoint SAM3 nằm ngoài `weights/` |
| `--table-only` | chỉ tạo lại `outputs/ablation_table.csv` |

Từng bước cũng chạy riêng được, ví dụ:

```bash
python sam3_baseline_<ds>.py --arm llm
python val_unetasppdinov2_<ds>.py --arm llm
```

---

## 5. PC59: PASCAL-Context 59

### 5.1 Dữ liệu: `pc59/data/`

Tải tại https://drive.google.com/drive/folders/1dCwcG-Pyh_FwatlaPVYjBrcNs2yUwGWk?usp=sharing

```
pc59/data/
├── JPEGImages/                 # ảnh VOC2010: <id>.jpg
├── SegmentationClassContext/   # mask PNG: 0 = nền, 1..59 = lớp theo alphabet (tools/prepare_pc59_mat_to_png.ipynb)
├── pascal_context_train.txt    # 4998 dòng "JPEGImages/<id>.jpg GroundTruth_trainval_png/<id>.png"
└── pascal_context_val.txt      # 5105 dòng
```

- Đường dẫn khai báo trong mục A của `pc59_common.py` (`DATA_DIR`, `IMAGES_DIR`...). Nếu dữ liệu nằm chỗ khác thì sửa ở đó.
- `59_labels.txt` **không cần**. Thứ tự 59 lớp được viết cứng theo alphabet trong `pc59_common.py`.

### 5.2 Configs có sẵn

| File | Nội dung |
|---|---|
| `target_classes.json` | 34 lớp (select_refinement_classes.py: GMM, ΔBIC = 6.34) |
| `contexts_class_pc59.json` | ngữ cảnh viết tay cho LLM |
| `adjust_prompt_pc59.json` | prompt LLM **thật**, ví dụ `shelves -> shelves, shelf, bookshelf, rack` |



### 5.4 Chạy (nhánh no-LLM đã xong, chỉ chạy nhánh LLM)

```bash
cd pc59
python run_pipeline_pc59.py --skip-nollm --dry-run
python run_pipeline_pc59.py --skip-nollm --limit 5
python run_pipeline_pc59.py --skip-nollm
```

### 5.5 Quy ước metric (giữ nguyên như bản đã chạy)

- Nhãn eval: 0..58 là lớp, 255 là ignore. Pixel nền (raw 0) bị bỏ qua.
- Pixel mà SAM3 không gán cho lớp nào mang giá trị 255 và **không** được tính vào confusion matrix.
- mIoU là trung bình trên đủ 59 lớp.
- Hyperparameter: ảnh 512×512, batch 32, patience 8. Tversky (0.5, 0.5) và λ = 1 cho mọi lớp, không có CE boost (giống bản no-LLM đã chạy).

### 5.6 Kiểm tra riêng cho PC59

```bash
python tools/check_pc59_class_order.py      # 59 lớp alphabet, offset raw/canonical, vài mask GT thật
```

---

## 6. VOC: Pascal VOC 2012

### 6.1 Dữ liệu: `voc/data/VOC2012/`

download: https://drive.google.com/file/d/10Dw_dc4xiMh6AxcABGJXvVLOyRv82M4H/view?usp=sharing


Dùng bộ **VOC2012** chuẩn (`VOCtrainval_11-May-2012.tar`, hoặc bộ đã dùng trên Kaggle). Chỉ cần 3 thư mục con:

```
voc/data/VOC2012/
├── JPEGImages/                         # <id>.jpg
├── SegmentationClass/                  # mask PNG (palette): 0 = nền, 1..20 = lớp, 255 = viền/void
└── ImageSets/
    └── Segmentation/
        ├── train.txt                   # 1464 id, mỗi dòng 1 id (vd. 2007_000032)
        └── val.txt                     # 1449 id
```



Lưu ý:
- Dùng `SegmentationClass/`, **không** dùng `SegmentationObject/`.
- **Không** dùng VOC2010 của PC59: VOC cần đúng split segmentation của VOC2012.

### 6.2 Configs có sẵn

| File | Nội dung |
|---|---|
| `target_classes.json` | 6 lớp: tvmonitor, chair, bicycle, sofa, pottedplant, diningtable (cùng danh sách và thứ tự với notebook cũ). Nếu có file gốc từ `select_refinement_classes.py` thì thay vào |
| `contexts_class_voc.json` | ngữ cảnh viết tay |
| `adjust_prompt_voc.json` | prompt LLM thật, ví dụ `tvmonitor -> monitor, computer screen` |

### 6.3 Chạy (cả 2 nhánh, skip nollm để chạy llm)

```bash
cd voc
python run_pipeline_voc.py --dry-run
python run_pipeline_voc.py --limit 5

python run_pipeline_voc.py --skip-nollm
```

Muốn chạy từng nhánh riêng thì thêm `--skip-llm` hoặc `--skip-nollm`.

### 6.4 Quy ước metric và hyperparameter (giống notebook cũ)

- Nhãn eval: 0 = nền, 1..20 = lớp, 255 = void (bỏ qua).
- Pixel SAM3 không gán lớp nào được tính là **nền (0)**, nên pixel bị bỏ sót vẫn làm giảm IoU của lớp đó.
- mIoU là trung bình trên **20 lớp** (không tính nền).
- Hyperparameter: ảnh 512×512, batch 32, patience 10, CE boost ×1.5 cho `bicycle`. Tversky/Boundary theo từng lớp:
  - sofa 0.5/0.5, λ 1.0
  - chair 0.45/0.55, λ 1.5
  - diningtable 0.45/0.55, λ 1.0
  - tvmonitor 0.40/0.60, λ 2.0
  - pottedplant 0.35/0.65, λ 2.0
  - bicycle 0.35/0.65, λ 2.5
- Các giá trị này nằm trong `TRAIN_PARAMS`, mục A của `voc_common.py`.

---

## 7. Cityscapes

### 7.1 Dữ liệu: `cityscapes/data/cityscapes/`

download: https://drive.google.com/file/d/1vEu15A3In1CQ00hFFzgpHCEIR1kWHzjx/view?usp=sharing

Cần 2 gói chính thức: `leftImg8bit_trainvaltest.zip` và `gtFine_trainvaltest.zip`. Cấu trúc cần có:

```
cityscapes/data/cityscapes/
├── leftImg8bit/
│   ├── train/<city>/<city>_xxxxxx_yyyyyy_leftImg8bit.png   # 2975 ảnh, 18 thành phố
│   └── val/<city>/...                                      # 500 ảnh (frankfurt, lindau, munster)
└── gtFine/
    ├── train/<city>/<city>_xxxxxx_yyyyyy_gtFine_labelIds.png
    └── val/<city>/...
```

- Mỗi ảnh chỉ cần file `*_gtFine_labelIds.png`. Pipeline tự đổi labelId sang trainId bằng bảng chuyển đổi chính thức.
- Nếu có sẵn `*_gtFine_labelTrainIds.png` thì pipeline dùng file đó.
- Thư mục `test/` không cần.

Sắp xếp từ bộ dữ liệu cũ, dạng `data_cityscapes/leftImg8bit_trainvaltest/leftImg8bit/...`:

```bash
cd cityscapes && mkdir -p data/cityscapes
mv <data_cityscapes>/leftImg8bit_trainvaltest/leftImg8bit data/cityscapes/leftImg8bit
mv <data_cityscapes>/gtFine_trainvaltest/gtFine           data/cityscapes/gtFine
```

### 7.2 Configs có sẵn

| File | Nội dung |
|---|---|
| `target_classes.json` | 14 lớp: sidewalk, bus, vegetation, bicycle, motorcycle, person, traffic sign, traffic light, pole, fence, wall, train, rider, terrain (cùng danh sách và thứ tự với notebook cũ) |
| `contexts_class_city.json` | ngữ cảnh viết tay |
| `adjust_prompt_city.json` | prompt LLM thật (bản v3), ví dụ `person -> pedestrian, child ` |

### 7.3 Chạy (cả 2 nhánh, skip nollm để chạy llm)

```bash
cd cityscapes
python run_pipeline_cityscapes.py --dry-run
python run_pipeline_cityscapes.py --limit 5

python run_pipeline_cityscapes.py --skip-nollm
```



### 7.4 Quy ước metric và hyperparameter (giống notebook cũ)

- Nhãn eval: trainId + 1 (1..19 là lớp), 0 = "unlabeled", 255 = ignore.
- Pixel SAM3 không gán lớp nào được tính là **unlabeled (0)**, nên pixel bị bỏ sót vẫn làm giảm IoU.
- mIoU là trung bình trên **19 lớp**.
- Hyperparameter: ảnh 512×1024, batch 32, patience 8, CE boost ×1.5 cho `rider`. Tversky/Boundary theo từng lớp:
  - (0.35, 0.65) cho các lớp nhỏ và mảnh: bicycle, motorcycle, traffic sign, traffic light, rider;
  - λ = 2.5 cho pole, traffic sign, traffic light.
- Chi tiết nằm trong `TRAIN_PARAMS`, mục A của `cityscapes_common.py`.

---

## 8. Kết quả (giống nhau cho mọi dataset)

```
<ds>/outputs/
├── split.json                    # split train/val nội bộ (dùng chung 2 nhánh)
├── dinov2_cache/                 # dùng chung 2 nhánh
├── baseline_nollm/               # (tùy chọn, tự copy vào) baseline no-LLM cũ
├── nollm/                        # cấu trúc giống llm/
├── llm/
│   ├── baseline/                 # SAM3 baseline với prompt của nhánh này
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

- `summary_metrics.csv`: Pixel Accuracy, mIoU, Mean Dice, **mIoU_w** (lớp mục tiêu), **mIoU_s** (các lớp còn lại).
- `per_class_metrics.csv`: IoU, Dice, GT Pixels, Pred Pixels của từng lớp. File val có thêm cột `Source` cho biết lớp đó lấy từ `SAM3 + <kiến trúc> (hybrid)` hay `SAM3 (baseline)`.
- `per_class_iou_bar_chart.png`.
- `class_visualizations/`: 2 ảnh mẫu cho mỗi lớp.
- `manifest.json`: fingerprint, prompt đã dùng, số ảnh.

`ablation_table.csv` có mỗi dòng là một cấu hình (baseline hoặc hybrid × nhánh × kiến trúc), với các cột PA, mIoU, mDice, mIoU_w, mIoU_s.

Với mọi dataset:
- Một mask tham gia cạnh tranh giữa các lớp khi score > 0.30.
- Mask thô dùng union-at-threshold với các ngưỡng [0.5, 0.3, 0.2, 0.15].
- Hàng rỗng "background" hoặc "unlabeled" không nằm trong bảng per-class.

---

## 9. Kiểm tra với checkpoint cũ

Chạy val hybrid mới trên một checkpoint cũ (cần `outputs/<arm>/val_cache` do `sam3_baseline_<ds>.py --arm <arm>` tạo ra):

```bash
python val_unetaspp_pc59.py --arm nollm --ckpt /path/weights_unetaspp_pc59_nollm_v1/unetaspp_pc59_nollm_v1_best.pth
# -> outputs/nollm/unetaspp/val_external/, số liệu phải khớp với kết quả val cũ
```

---

## 10. Lỗi thường gặp

| Thông báo | Cách xử lý |
|---|---|
| `[MISSING] .../data/...` trong preflight | Dữ liệu chưa đặt đúng chỗ. Đối chiếu với sơ đồ `data/` của dataset (mục 5.1 / 6.1 / 7.1) |
| `LLM arm: ... is a PLACEHOLDER` | `configs/adjust_prompt_<ds>.json` là file giữ chỗ. Thay bằng file thật |
| `classes in adjust_prompt_<ds>.json do not match target_classes.json` | Hai file lệch nhau. Sửa cho khớp, hoặc sinh lại prompt bằng `tools/generate_adjust_prompt_<ds>.py` |
| `split.json does not match the train list` | `outputs/split.json` không phải của dataset này, hoặc danh sách train đã đổi. Xóa đi để pipeline tạo lại (seed 42) |
| `[stale] ... moved ... .stale-<time>` | Output cũ được build với cấu hình khác nên đã được chuyển sang tên `*.stale-*`. Xóa khi không cần nữa |
| `DINOv2 unavailable` | Server không có mạng. Đặt repo và trọng số DINOv2 vào `weights/dinov2-repo/` và `weights/dinov2_vits14_pretrain.pth` |
| CUDA out of memory khi train Cityscapes | Giảm `batch_size` trong `TRAIN_PARAMS` (`cityscapes_common.py`), rồi chạy lại lệnh cũ. Đổi hyperparameter sẽ đổi fingerprint, nên bước train tự chạy lại, không cần `--force` |

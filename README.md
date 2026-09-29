# Fundus Gate: Bộ Lọc Phân Loại Ảnh Đáy Võng Mạc Mắt (Input Validation Gate)

Mô hình phân loại ảnh nhị phân (**Binary Image Classifier**) đóng vai trò như một **Cổng kiểm duyệt đầu vào (Input Validation Gate)** trước khi chuyển ảnh vào mô hình phân loại bệnh võng mạc tiểu đường (**Diabetic Retinopathy - DR Classifier**) của khóa luận tốt nghiệp:
*“Nghiên cứu Vision Transformer và Ứng dụng Phân loại Mức độ Bệnh Tiểu đường từ Ảnh Đáy Võng mạc Mắt”* (HUIT, 09/2026).

```text
       Người dùng tải ảnh lên
                 ↓
      ┌─────────────────────┐
      │     FUNDUS GATE     │
      │  (MobileNetV3Small) │
      └──────────┬──────────┘
                 │
      ┌──────────┴──────────┐
      ▼                     ▼
[ FUNDUS (1) ]       [ NON-FUNDUS (0) ]
      │                     │
      ▼                     ▼
Chuyển sang           Từ chối (Reject)
DR Classifier         Yêu cầu chụp lại
(ResNet50 / ViT-B16)  ảnh đáy mắt hợp lệ
```

---

## 1. Mục tiêu và Nguyên lý Hoạt động

* **Mục tiêu duy nhất:** Phát hiện chính xác ảnh đáy võng mạc mắt (Retinal Fundus Photograph) và từ chối mọi loại ảnh không hợp lệ (ảnh sinh hoạt, động vật, phong cảnh, ảnh y tế khác như X-ray/MRI/OCT, và đặc biệt là ảnh chụp mắt thông thường/mống mắt/bề mặt nhãn cầu).
* **Quy ước nhãn:**
  * `0 = non_fundus` (Không phải ảnh đáy võng mạc)
  * `1 = fundus` (Ảnh đáy võng mạc hợp lệ)
* **Nguyên tắc phân tầng:** Mô hình cổng này **tuyệt đối không tham gia chẩn đoán mức độ DR (grade 0–4)**. Khi qua cổng với nhãn `fundus`, ảnh mới được đưa vào mô hình CNN / Vision Transformer chuyên biệt để phân loại bệnh.

---

## 2. Kiến trúc Mô hình (Model Architecture)

Thiết kế ưu tiên tiêu chí **kích thước siêu nhỏ, tốc độ suy luận mili-giây, tiêu tốn ít bộ nhớ**, tối ưu để tích hợp trực tiếp trên web client, trình duyệt hoặc thiết bị di động:

* **Backbone:** `MobileNetV3Small` (Pretrained trên ImageNet-1K).
* **Input Resolution:** `224 × 224 × 3` (RGB).
* **Classification Head:**
  * `GlobalAveragePooling2D`
  * `Dropout(rate=0.3)`
  * `Dense(1, activation='sigmoid', dtype='float32')`
* **Tham số mô hình:** 939,697 tham số (939.7K params).
* **Kích thước file:**
  * Keras (`fundus_gate.keras`): **9.78 MB**
  * TFLite Float32 (`fundus_gate.tflite`): **3.56 MB**
  * TFLite Float16 (`fundus_gate_float16.tflite`): **1.82 MB**

---

## 3. Dữ liệu Huấn luyện & Các Nhóm Negative Bắt buộc

Tổng số lượng ảnh được cân bằng tỷ lệ **50% Fundus / 50% Non-Fundus** (~21.000 ảnh).

### 3.1. Nhóm Positive: Fundus (`label = 1`)
* **Nguồn:** Tập dữ liệu **EyePACS** (`diabetic-retinopathy-resized`).
* **Số lượng:** 10.500 ảnh đáy võng mạc mắt chuẩn y khoa (8.952 bệnh nhân duy nhất).
* **Chống rò rỉ (Zero Leakage):** Sử dụng `GroupShuffleSplit` theo mã bệnh nhân (`patient_id` tách từ tên file `<patient_id>_left/right`). Đảm bảo hai mắt của cùng một bệnh nhân **tuyệt đối không bao giờ xuất hiện đồng thời ở cả tập Train và Val/Test** (đã kiểm định bằng assert giao tập rỗng).

### 3.2. Nhóm Negative: Non-Fundus (`label = 0`)
Được thiết kế cân bằng chặt chẽ qua 3 phân nhóm:

#### A. Generic Negatives (3.500 ảnh)
* **Natural Images (`prasunroy/natural-images`):** Xe cộ, động vật, con người, hoa quả, tĩnh vật.
* **Intel Image Classification (`puneet6060/intel-image-classification`):** Phong cảnh tự nhiên, rừng, núi, biển, đường phố, tòa nhà.
* **Document Classification (`ritvik1909/document-classification-dataset`):** Tài liệu scan, hóa đơn, email, văn bản chụp màn hình.

#### B. Medical Negatives (3.500 ảnh)
* **Retinal OCT (`shakilrana/octdl-retinal-oct-images-dataset`):** Chụp cắt lớp quang học võng mạc (Optical Coherence Tomography). *Đây là nhóm y tế ưu tiên hàng đầu vì cùng thuộc chuyên khoa mắt nhưng không phải ảnh màu đáy mắt.*
* **Chest X-Ray (`thomasdubail/chest-pneumonia-256x256`):** Ảnh chụp X-quang lồng ngực.
* **Brain MRI (`sartajbhuvaji/brain-tumor-classification-mri`):** Ảnh cộng hưởng từ sọ não.

#### C. Ocular Hard Negatives (3.500 + 900 Challenge ảnh)
*Nhóm thử thách quan trọng nhất: Chứa các cấu trúc liên quan đến mắt nhưng không phải đáy võng mạc:*
* **Iris Patterns (`monareyhanii/casia-iris-syn`):** Ảnh cận cảnh mống mắt, đồng tử, vân mống mắt độ phân giải cao.
* **Eye Close-ups (`kayvanshah/eye-dataset`):** Ảnh chụp mắt người (mắt nhắm, mở, mi mắt, lông mi).
* **Anterior Segment & Slit-lamp (`akshayramakrishnan28/cataract-classification-dataset`, `kondwani/eye-disease-dataset`):** Ảnh bán phần trước nhãn cầu, khám đèn khe (slit-lamp), đục thủy tinh thể, viêm kết mạc.
* **Tách riêng Challenge Test Set:** Đã trích lập riêng **900 ảnh Ocular Hard Negatives** làm tập kiểm thử thử thách độc lập. Mô hình **hoàn toàn không được nhìn thấy các ảnh này trong suốt quá trình huấn luyện và chọn ngưỡng (Zero Exposure)**.

---

## 4. Phân chia Dữ liệu Thực tế (Dataset Split)

| Phân vùng | Fundus (1) | Generic (0) | Medical (0) | Ocular Hard (0) | Tổng số mẫu | Tỷ lệ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train Set** | 7.354 | 2.450 | 2.450 | 2.449 | **14.703** | **70%** |
| **Validation Set** | 1.558 | 525 | 525 | 525 | **3.133** | **15%** |
| **Test Set** | 1.588 | 525 | 525 | 526 | **3.164** | **15%** |
| **Challenge Test Set** | 0 | 0 | 0 | **900** | **900** | *Độc lập* |

---

## 5. Quy trình Huấn luyện (Training Pipeline)

Huấn luyện hai giai đoạn (Two-Stage Transfer Learning) với `tf.data`, RAM Caching và Light Augmentation:

* **Stage 1 (Frozen Backbone - 3 Epochs):**
  * Đóng băng toàn bộ các lớp của MobileNetV3Small.
  * Chỉ cập nhật Classification Head (`GAP -> Dropout(0.3) -> Dense(1, sigmoid)`).
  * Optimizer: `Adam(lr=1e-3)`, Loss: `BinaryCrossentropy()`.
* **Stage 2 (Fine-tuning Top 30% Layers - 3 Epochs):**
  * Mở đóng băng 48 lớp trên cùng (~30% layers).
  * Optimizer: `Adam(lr=1e-4)` kết hợp `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`.
  * Augmentation nhẹ: `RandomFlip("horizontal")`, `RandomRotation(0.05)`, `RandomZoom(0.05)`, `RandomContrast(0.1)`.

---

## 6. Tối ưu hóa Ngưỡng Quyết định (Validation-Calibrated Threshold)

* Quét lưới 197 mốc ngưỡng $\tau \in [0.01, 0.99]$ trên **tập Validation**:
  $$\text{Điều kiện ràng buộc: } \text{Recall}_{\text{fundus}} \ge 95.0\%$$
  $$\text{Hàm mục tiêu: } \min (\text{False Positive Rate}) \iff \max (\text{Specificity})$$
* **Ngưỡng tối ưu thực nghiệm được chọn:** $\tau^* = \mathbf{0.9650}$ (được lưu tại `threshold.json`).
* **Validation Metrics tại ngưỡng $\tau^* = 0.9650$:**
  * Recall: **100.00%**
  * Specificity: **100.00%**
  * FPR: **0.00%**
  * F1-Score: **1.0000**
  * ROC-AUC: **1.0000**

---

## 7. Đánh giá Chi tiết & Báo cáo Thử nghiệm

### 7.1. Kết quả trên Tập Kiểm thử Chuẩn (Test Set - 3,164 ảnh)
* **Accuracy:** **99.94%**
* **Precision:** **99.87%**
* **Recall (Sensitivity):** **100.00%** (1,588/1,588 ảnh Fundus được nhận diện chính xác, **0 ca bỏ sót**)
* **Specificity (TNR):** **99.87%** (1,574/1,576 ảnh Non-Fundus bị từ chối chính xác)
* **False Positive Rate (FPR / False Acceptance):** **0.13%** (chỉ 2 ảnh Non-Fundus bị chấp nhận nhầm)
* **False Negative Rate (FNR):** **0.00%**
* **F1-Score:** **0.9994**
* **ROC-AUC:** **1.0000**
* **Confusion Matrix:** $\text{TP} = 1588, \text{FP} = 2, \text{TN} = 1574, \text{FN} = 0$

### 7.2. Kết quả trên Ocular Hard Negatives Challenge Test Set (900 ảnh cô lập)
* **Tổng số mẫu thử thách:** 900 ảnh (Mống mắt CASIA, cận cảnh nhãn cầu, đục thủy tinh thể, bệnh mắt ngoài).
* **Tỷ lệ từ chối thành công (Challenge Specificity):** **99.89%** (899/900 ảnh bị loại bỏ chính xác).
* **Tỷ lệ chấp nhận nhầm (Challenge FPR):** **0.11%** (duy nhất 1 ảnh bị lọt qua).

---

## 8. Hướng dẫn Triển khai & Sử dụng CLI (`predict.py`)

### 8.1. Cài đặt môi trường
Chỉ yêu cầu thư viện siêu nhẹ (`tflite_runtime` hoặc `tensorflow`, kèm `Pillow`, `numpy`):

```bash
pip install numpy pillow tflite-runtime
```

### 8.2. Chạy suy luận (Inference)

Mô hình tự động đọc ngưỡng tối ưu từ `threshold.json`:

```bash
python predict.py --image path/to/sample_fundus.jpg
```

**Kết quả khi ảnh là đáy võng mạc hợp lệ:**
```text
Probability: 0.98
Threshold: 0.62
Decision: ACCEPT
```

**Kết quả khi ảnh là ngoại cảnh, X-ray hoặc mắt thường:**
```bash
python predict.py --image path/to/cat_or_iris.jpg
```
```text
Probability: 0.03
Threshold: 0.62
Decision: REJECT
```

Tùy chọn chỉ định mô hình và ngưỡng thủ công:
```bash
python predict.py --image test.jpg --model fundus_gate_float16.tflite --threshold 0.70
```

---

## 9. Phân tích Sai số (Error Analysis)

Các trường hợp có nguy cơ gây nhầm lẫn và cách khắc phục:
1. **Ảnh đáy mắt thiếu sáng cực độ hoặc mất nét nghiêm trọng:** Có thể bị dự đoán $P < \tau^*$ (False Negative). Đây là hành vi mong muốn trong thực tế lâm sàng vì ảnh quá mờ không thể dùng để chẩn đoán DR.
2. **Ảnh chụp cận cảnh đồng tử giãn (mydriasis) với phản chiếu ánh hồng:** Mô hình MobileNetV3Small với các đặc trưng mạch máu học được từ EyePACS nhận diện tốt sự khác biệt giữa cấu trúc gai thị/mạch máu võng mạc và đồng tử thường.

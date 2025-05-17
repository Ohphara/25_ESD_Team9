# Vision-Based Walking Assistant for Visually Impaired Pedestrians

## Introduction

### Background

* Visually impaired pedestrians rely solely on auditory cues and canes to perceive their surroundings, making them more vulnerable to traffic and safety hazards.
* Survey results show that the primary barriers to outdoor mobility are physical discomfort (25.9%) and lack of assistance (33.4%), highlighting the need for improved accessibility and support services.
* Obstacles such as personal mobility devices blocking tactile paving further impair safe navigation.

### Project Objective

* **System Latency Requirement**: End-to-end latency must not exceed **100 ms** (SAE J2945 standard).
* **Inference Time Budget**: Model inference must be under **66 ms** (≈15 FPS) to allow time for preprocessing and warning transmission.
* **Goal**: Develop a real-time vision-based system that detects nearby hazards, analyzes risk, and delivers vibration and audio warnings to visually impaired users.

## Training

We trained a YOLOv11n model on the AIHub Road Obstacle Dataset to detect various real-world obstacles in outdoor environments.

### 1. Dataset

- **Source**: [AIHub Obstacle Detection Dataset](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=189)
- Includes annotated images of various road conditions and obstacles
- Converted to YOLO format using custom preprocessing scripts

### 2. Model and Framework

- **Base Model**: YOLOv11n (Ultralytics latest)
- **Training Framework**: [Ultralytics `ultralytics` Python package](https://docs.ultralytics.com)
- **Environment**: Google Colab (GPU enabled)
- **Transfer Learning**: Fine-tuned from pretrained checkpoint `final_best3.pt`

### 3. Training Configuration

| Parameter         | Value              |
|------------------|--------------------|
| Epochs           | 30                 |
| Learning Rate    | 0.001              |
| Batch Size       | Default (Auto)     |
| Image Size       | 640x640 (default)  |
| Dataset YAML     | `yolodata/data.yaml` |
| Optimizer        | Default (SGD/Adam) |

> All other hyperparameters were left as default values used in the Ultralytics training loop.

### 4. Training Script

Training was performed on Google Colab using the following script:

```python
from ultralytics import YOLO
import shutil

# Load pretrained YOLOv11n model
model = YOLO('/content/drive/MyDrive/yolo_models/final_best3.pt')

# Train the model
model.train(
    data='yolodata/data.yaml',
    epochs=30,
    lr0=0.001
)

# Backup best weights to Google Drive
shutil.copy(
    '/content/runs/detect/train/weights/best.pt',
    '/content/drive/MyDrive/yolo_models/final_best4.pt'
)
### 5. Evaluation Results
The trained model was evaluated on 2,149 validation images with 20,437 total instances.

Metric	Value
Precision	0.690
Recall	0.433
mAP@0.5	0.528
mAP@0.5:0.95	0.344

✅ Strongly Performing Classes
Class	mAP@0.5	mAP@0.5:0.95
car	0.920	0.709
pole	0.845	0.605
tree_trunk	0.827	0.518
person	0.773	0.474
movable_signage	0.698	0.483

These classes had a sufficient number of samples and distinct features, resulting in strong detection performance.

⚠️ Underperforming Classes
Class	mAP@0.5	Samples	Notes
wheelchair	0.037	3	Too few samples
cat / dog	0.0~0.42	1~4	Rarely present
parking_meter	0.0	3	Nearly unrecognizable
chair / bench	~0.4	100~200	Visual ambiguity

Low-performing classes can be improved by data augmentation or considered for class merging or removal depending on deployment goals.
## Optimization

We optimize YOLO models on a Raspberry Pi 5 platform using **ONNX** and **NCNN** to meet latency requirements.

### 1. ONNX Optimization

* **Guide**: [https://docs.ultralytics.com/en/4.x/integrations/onnx/#supported-deployment-options](https://docs.ultralytics.com/en/4.x/integrations/onnx/#supported-deployment-options)
* **Steps**:

  1. Export PyTorch model to ONNX format
  2. Use ONNX Runtime (CPU) for inference acceleration

### 2. NCNN Optimization

* **Guide**: [https://docs.ultralytics.com/en/4.x/integrations/ncnn/#installation](https://docs.ultralytics.com/en/4.x/integrations/ncnn/#installation)
* **Findings**:

  * Enabling FP16 in NCNN degrades YOLO detection accuracy.
  * **Recommendation**: Use FP32 mode only (disable FP16).

### 3. NCNN with Vulkan Backend (Work in Progress)

* **Objective**: Leverage Raspberry Pi 5 GPU via Vulkan for further speedup.
* **Todo**:

  * Integrate Vulkan SDK in NCNN build
  * Test driver compatibility
  * Document usage steps

## Benchmark Results

Measured times for input size **(3, 640, 640)** on Raspberry Pi 5:

| Model          | Preprocess (ms) | Inference (ms) | Postprocess (ms) |
| -------------- | --------------: | -------------: | ---------------: |
| YOLO11n (ONNX) |            10.7 |          248.4 |             10.9 |
| YOLO11n (NCNN) |            10.8 |          113.6 |              4.0 |
| YOLOv8n (ONNX) |            10.3 |          251.8 |              7.8 |
| YOLOv8n (NCNN) |            10.7 |          114.6 |              4.0 |
| YOLOv5n (ONNX) |            10.5 |          216.8 |             11.0 |
| YOLOv5n (NCNN) |            10.9 |          110.7 |              4.0 |

## Team Responsibilities

* **Piljun**: Model Optimization, Monocular Depth Estimation
* **Hoyeon**: Model Transfer Learning, System Integration

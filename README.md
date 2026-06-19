# NUS-SOC

A repository for intelligent vision scenarios, with two implementations:

- `baseline`: Image classification and visualization client based on TensorFlow/Keras + MQTT + DearPyGui.
- `Advanced`: Multi-model real-time detection and state/direction publishing system based on YOLOv7 + SORT.

## Project Summary

This project implements a **real-time monitoring robot system based on YOLOv7** for specialized surveillance scenarios.
It focuses on multi-class behavior/object detection (e.g., smoking, fighting, falling, and littering), stable multi-target tracking,
and an end-to-end deployment pipeline connecting robots, edge devices, and cloud-side services.

Core engineering work includes:

- Multi-class dataset collection, cleaning, preprocessing, and YOLOv7 fine-tuning for improved detection accuracy and inference speed.
- A multi-model detection pipeline integrated with SORT to maintain stable target persistence in complex scenes.
- Continuous video behavior recognition through temporal decision logic over streaming frames.
- An End-Edge-Cloud architecture over MQTT for low-latency messaging between robot-side components and backend services.
- A DearPyGui monitoring client supporting the workflow from video input and model inference to real-time visualization.

## Project Structure

```text
NUS-SOC/
├── baseline/
│   ├── backend/                 # MQTT inference server + test sender/listener scripts
│   ├── client/                  # DearPyGui client (model switching, result browsing, status display)
│   ├── utils/                   # Training, data processing, and prediction scripts
│   └── requirements.txt
├── Advanced/
│   ├── advanced_final/          # Main multi-model real-time detection program (smoke/litter/fall/fight)
│   ├── dataset_preprocess/      # YOLO dataset merge tools and sample datasets
│   ├── docs/
│   └── yolov7/                  # YOLOv7 code and data
└── README.md
```

## Features Overview

### Baseline

- MQTT inference workflow: publish image payloads to `Group19/IMAGE/classify`, receive results from `Group19/IMAGE/predict`.
- Supports flower test classification and cat breed classification (requires local model files).
- DearPyGui client supports:
- Model switching and refresh
- Result image browsing and prediction display
- Cat status tracking panel
- Auto-archiving `results` and `result_image.json` on exit

### Advanced

- YOLOv7 multi-model parallel detection: `smoke` / `litter` / `fall` / `fight`
- SORT object tracking and direction decision (`left` / `right` / `no_action`)
- Supports local camera and Flask video stream (Raspberry Pi scenario)
- Publishes state and direction via MQTT:
- `Group19/CONTROL/state`
- `Group19/CONTROL/direction`

## Environment Requirements

- Python 3.9+ (3.10 recommended)
- An available MQTT broker (default port `1883`)
- A camera device (for running `Advanced`)
- Optional GPU (`Advanced` performs better with CUDA)

## Baseline Quick Start

### 1) Install Dependencies

Run from the project root:

```bash
cd baseline
pip install -r requirements.txt
pip install dearpygui pillow
```

### 2) Prepare MQTT Credential File

Both `baseline/backend` and `baseline/client` read `mqtt.pwd` in this format (username and password on one line, separated by a space):

```text
username password
```

### 3) Prepare Model Files (Important)

Large model files are ignored in this repo (e.g., `client/model/*` in `.gitignore`).  
Place trained models in the paths below (filenames must match the code):

- `baseline/client/model/cats_efficientnetb0-Noise-Brightness-V1.keras`
- `baseline/client/model/cat_classifier_xception.h5`
- `baseline/client/model/cats_insection.keras`
- `baseline/client/model/cats_efficientnetb0-Noise-Brightness-V3-bright-05.keras`
- `baseline/client/model/cats_efficientnetb0-Noise-Brightness-V3-dark-05.keras`
- `baseline/client/model/cats_matching.keras`
- `baseline/client/model/flowers.keras` (test model)

### 4) Start Client (With Built-in Inference Service)

```bash
cd baseline/client
python app.py
```

Notes:

- The client initializes `MQTTInferenceServer` when importing `system`.
- The default broker address is hardcoded in `baseline/client/system/server_sys.py`. For local setup, change it to `127.0.0.1`.

### 5) Optional: Start Backend Scripts Separately

```bash
cd baseline/backend
python server.py      # Inference service
python carsender.py   # Batch-send test images
python monitor.py     # Listen for prediction results only
```

## Advanced Quick Start

### 1) Install Dependencies

```bash
cd Advanced/advanced_final
pip install -r requirements.txt
pip install paho-mqtt
```

### 2) Prepare Model Weights

Default model paths:

- `trained_model/smoke.pt`
- `trained_model/new_trash_best.pt`
- `trained_model/fall.pt`
- `trained_model/fight.pt`

### 3) Prepare MQTT Credential File

Create `mqtt.pwd` in `Advanced/advanced_final`:

```text
username password
```

### 4) Start Multi-model Detection + MQTT Publishing

```bash
cd Advanced/advanced_final
python main.py --camera 0 --device cpu --mqtt-host 127.0.0.1
```

Arguments:

- `--camera 0`: local camera
- `--camera 1`: Flask video stream (default URL in `run_multi_model.py` as `FLASK_URL`)
- `--conf-thres`: confidence threshold (default `0.5`)
- `--iou-thres`: NMS threshold (default `0.45`)
- `--device`: `cpu` or GPU index (e.g., `0`)
- `--mqtt-host`: MQTT broker address

## YOLO Dataset Merge Tool

`Advanced/dataset_preprocess/YoloDatasetMerger.py` merges multiple YOLO-format datasets and automatically resolves class IDs.

Example:

```bash
cd Advanced/dataset_preprocess
python YoloDatasetMerger.py --datasets stand_fall_dataset violence3 --output merged_dataset
```

## MQTT Topic Conventions

### Baseline

- Publish images for classification: `Group19/IMAGE/classify`
- Publish classification results: `Group19/IMAGE/predict`

### Advanced

- Subscribe control status: `Group19/CONTROL`
- Publish state: `Group19/CONTROL/state`
- Publish direction: `Group19/CONTROL/direction`

## Troubleshooting

- Model not found at startup: verify filenames in `baseline/client/model` or `Advanced/advanced_final/trained_model`.
- MQTT connection failed: check `mqtt.pwd` format, broker address, port `1883`, and account permissions.
- Client failed to launch: ensure `dearpygui` and `tensorflow` are installed.
- Slow `Advanced` inference: prefer GPU, reduce input resolution, or increase frame interval.

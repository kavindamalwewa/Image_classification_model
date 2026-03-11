<div align="center">

# 🚗 Vehicle Image Classification Model

### A CNN-based deep learning model that classifies vehicle images into 4 categories

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-CNN-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Overview

This project trains a **Convolutional Neural Network (CNN) from scratch** to classify vehicle images into 4 categories. It includes a full training pipeline with data augmentation, evaluation metrics, and a **GUI-based prediction tool** built with Tkinter and Matplotlib.

---

## 🏷️ Classes

| Icon | Label | Description |
|------|-------|-------------|
| 🚌 | `bus` | Public / private buses |
| 🚗 | `car` | Passenger cars & sedans |
| 🏍️ | `motorcycle` | Motorcycles & scooters |
| 🚚 | `truck` | Trucks & heavy vehicles |

---

## 🗂️ Project Structure

```
Image_classification_model/
│
├── 📂 Dataset/                      # Training data (organized by class folder)
│   ├── bus/
│   ├── car/
│   ├── motorcycle/
│   └── truck/
│
├── 🧠 train.py                      # Implementation 1 — Custom CNN from scratch
├── 🧠 implementation_2.py           # Implementation 2 — MobileNetV2 transfer learning
├── 🖼️  predict_gui.py                # GUI-based image prediction tool (Impl. 1)
├── 🖼️  implementation_2_predict.py   # GUI-based image prediction tool (Impl. 2)
├── 📊 training_history.py           # Plot training accuracy & loss graphs
│
├── 🏷️  labels.json                   # Class label mapping
├── 📈 training_history.json         # Saved training metrics per epoch
├── 💾 vehicle_model_cnn.h5          # Trained model — Impl. 1 (see download below)
├── 💾 model_v2.h5                   # Trained model — Impl. 2 (see download below)
│
└── 📄 README.md
```

---

## 🧠 Model Architecture

### Implementation 1 — Custom CNN (from scratch)

A custom **CNN built from scratch** — no transfer learning.

```
Input (160×160×3)
       │
  ┌────▼────┐
  │ Block 1 │  Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
  └────┬────┘
  ┌────▼────┐
  │ Block 2 │  Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
  └────┬────┘
  ┌────▼────┐
  │ Block 3 │  Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
  └────┬────┘
       │  Flatten
  ┌────▼────┐
  │Classifier│  Dense(256) → BatchNorm → Dropout(0.5)
  └────┬────┘
       │
  Dense(4) → Softmax
```

| Parameter | Value |
|-----------|-------|
| Input size | 160 × 160 × 3 |
| Conv blocks | 3 (32 → 64 → 128 filters) |
| Classifier | Dense(256) |
| Output | Dense(4) + Softmax |
| Optimizer | Adam (lr = 0.001) |
| Loss | Categorical Crossentropy |
| Epochs | 15 |
| Batch size | 32 |
| LR Scheduler | ReduceLROnPlateau (factor=0.3, patience=3) |

---

### Implementation 2 — MobileNetV2 Transfer Learning

Uses **MobileNetV2** pretrained on ImageNet as a frozen feature extractor, with a custom classification head.

```
MobileNetV2 (frozen, ImageNet weights)
       │
  GlobalAveragePooling2D
       │
  BatchNormalization
       │
  Dense(256, ReLU) → Dropout(0.5)
       │
  Dense(4) → Softmax
```

| Parameter | Value |
|-----------|-------|
| Base model | MobileNetV2 (frozen) |
| Input size | 160 × 160 × 3 |
| Classifier | Dense(256) |
| Output | Dense(4) + Softmax |
| Optimizer | Adam (lr = 0.0003) |
| Loss | Categorical Crossentropy |
| Epochs | up to 25 (EarlyStopping) |
| Batch size | 32 |
| Callbacks | ReduceLROnPlateau, EarlyStopping, ModelCheckpoint |

---

## 🔄 Data Augmentation

Training images are augmented on-the-fly to improve generalization:

- ↻ Rotation up to **20°**
- 🔍 Zoom up to **20%**
- ↔️ Width & height shift up to **10%**
- 🔁 Horizontal flip
- 📐 80/20 train-validation split

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/HashTag/Image_classification_model.git
cd Image_classification_model
```

### 2. Install dependencies

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn
```

### 3. Prepare your dataset

Organize the `Dataset/` folder with one sub-folder per class:

```
Dataset/
├── bus/
├── car/
├── motorcycle/
└── truck/
```

---

## 🚀 Usage

### 🏋️ Train the Model

#### Implementation 1 — Custom CNN

```bash
python train.py
```

This will:
- Load and augment training images
- Train the custom CNN for 15 epochs
- Save the trained model as `vehicle_model_cnn.h5`
- Save training history to `training_history.json`
- Display a confusion matrix and classification report
- Plot accuracy & loss curves

#### Implementation 2 — MobileNetV2 Transfer Learning

```bash
python implementation_2.py
```

This will:
- Load and augment training images
- Fine-tune MobileNetV2 for up to 25 epochs (with early stopping)
- Save the best checkpoint as `best_vehicle_model.h5` and final model as `model_v2.h5`
- Save training history to `training_history.json`
- Display a confusion matrix, classification report, and prediction grid
- Plot accuracy & loss curves

---

### 🖼️ Predict with GUI

#### Implementation 1

```bash
python predict_gui.py
```

#### Implementation 2

```bash
python implementation_2_predict.py
```

A file picker dialog will open — select any vehicle image. The tool will display:

- The uploaded image
- A **confidence bar chart** for all 4 classes
- **Top 3 predictions** with percentages in the terminal

---

### 📊 Plot Training History

```bash
python training_history.py
```

Generates side-by-side plots of:
- Training vs Validation **Accuracy**
- Training vs Validation **Loss**

---

## 💾 Downloads

### Dataset

The vehicle image dataset is hosted on Google Drive.

📥 **[Download Dataset from Google Drive](https://drive.google.com/file/d/1aqCxrwoU7-w5CO1k9mYcU2uLFP0TuyUy/view?usp=sharing)**

After downloading, extract and organize it as:

```
Image_classification_model/
└── Dataset/
    ├── bus/
    ├── car/
    ├── motorcycle/
    └── truck/
```

### Trained Model

The trained model file is too large to store on GitHub.

📥 **[Download `vehicle_model_cnn.h5` from Google Drive](https://drive.google.com/file/d/1ovi_R_7v4_eLFCxvnkcxRyGYcXeAbeQl/view?usp=drive_link)**

After downloading, place the file in the root project folder:

```
Image_classification_model/
└── vehicle_model_cnn.h5   ← place here
```

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `vehicle_model_cnn.h5` | Trained Keras model (Implementation 1) |
| `model_v2.h5` | Trained Keras model (Implementation 2) |
| `best_vehicle_model.h5` | Best checkpoint during Impl. 2 training |
| `training_history.json` | Per-epoch accuracy & loss values |
| `labels.json` | Class index → label mapping |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| ![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=tensorflow&logoColor=white&style=flat) | Deep learning framework |
| ![Keras](https://img.shields.io/badge/-Keras-D00000?logo=keras&logoColor=white&style=flat) | High-level neural network API |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white&style=flat) | Numerical computations |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?logo=python&logoColor=white&style=flat) | Plotting & visualization |
| ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white&style=flat) | Confusion matrix & metrics |
| ![Seaborn](https://img.shields.io/badge/-Seaborn-4C72B0?logo=python&logoColor=white&style=flat) | Heatmap visualization |
| ![Tkinter](https://img.shields.io/badge/-Tkinter-3776AB?logo=python&logoColor=white&style=flat) | GUI file picker dialog |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ using **TensorFlow** & **Keras**

</div>

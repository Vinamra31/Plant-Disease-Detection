# 🌿 Plant Disease Detection

A deep learning-based plant disease detection system that identifies diseases in pepper, potato, and tomato leaves. Built with a custom CNN trained from scratch on the PlantVillage dataset, served via a FastAPI backend and a Streamlit frontend.

---

## 📌 Overview

This project was built as part of a Computer Vision and Machine Learning lab. It takes an image of a plant leaf as input and predicts whether the leaf is healthy or diseased, along with the confidence score.

- **Model**: Custom CNN trained from scratch using PyTorch
- **Dataset**: PlantVillage (Pepper, Potato, Tomato — 15 classes, ~20,000 images)
- **Validation Accuracy**: ~92%
- **Backend**: FastAPI
- **Frontend**: Streamlit

---

## 🗂️ Project Structure

```
Plant Disease Detection/
├── model.pth           # Trained PyTorch model
├── classes.json        # Class index to label mapping
├── main.py             # FastAPI backend
├── app.py              # Streamlit frontend
├── run.bat             # Batch file to start both servers
└── README.md
```

---

## 🌱 Classes

The model can detect the following 15 conditions:

| Plant | Condition |
|-------|-----------|
| Pepper Bell | Bacterial Spot |
| Pepper Bell | Healthy |
| Potato | Early Blight |
| Potato | Late Blight |
| Potato | Healthy |
| Tomato | Bacterial Spot |
| Tomato | Early Blight |
| Tomato | Late Blight |
| Tomato | Leaf Mold |
| Tomato | Septoria Leaf Spot |
| Tomato | Spider Mites (Two-spotted) |
| Tomato | Target Spot |
| Tomato | Yellow Leaf Curl Virus |
| Tomato | Mosaic Virus |
| Tomato | Healthy |

---

## 🧠 Model Architecture

Custom CNN built with PyTorch:

```
Input (224x224x3)
    → Conv2D(32) + ReLU + MaxPool
    → Conv2D(64) + ReLU + MaxPool
    → Conv2D(128) + ReLU + MaxPool
    → Flatten
    → Linear(100352 → 128) + ReLU + Dropout(0.5)
    → Linear(128 → 15)
    → Softmax
```

**Training details:**
- Optimizer: Adam with L2 regularization (weight_decay=1e-4)
- Loss: CrossEntropyLoss
- Early Stopping: patience=3
- Image Size: 224x224
- Batch Size: 32
- Augmentation: Random horizontal flip, random rotation (20°)

---

## ⚙️ Setup & Installation

### Prerequisites
- Anaconda / Miniconda
- NVIDIA GPU (optional but recommended)
- Python 3.10+

### 1. Clone the repository
```bash
git clone https://github.com/Vinamra31/plant-disease-detection.git
cd plant-disease-detection
```

### 2. Create and activate conda environment
```bash
conda create -n ml_env python=3.10
conda activate ml_env
```

### 3. Install dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install fastapi uvicorn python-multipart streamlit requests pillow
```

### 4. Run the project
Either double-click `run.bat` or run manually:

```bash
# Terminal 1 - FastAPI backend
uvicorn main:app --reload

# Terminal 2 - Streamlit frontend
streamlit run app.py
```

---

## 🚀 Usage

1. Start both servers using `run.bat` or manually
2. Open `http://localhost:8501` in your browser
3. Upload a leaf image (JPG or PNG)
4. Click **Detect Disease**
5. View the predicted disease and confidence score

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/classes` | Get all 15 class labels |
| POST | `/predict` | Predict disease from uploaded image |

API docs available at `http://127.0.0.1:8000/docs`

### Example Request
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -F "file=@leaf_image.jpg"
```

### Example Response
```json
{
  "prediction": "Potato___Early_blight",
  "confidence": 99.83
}
```

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Validation Accuracy | 92% |
| Macro Avg Precision | 0.93 |
| Macro Avg Recall | 0.90 |
| Macro Avg F1-Score | 0.91 |

---

## 🛠️ Tech Stack

- **Model Training**: PyTorch, torchvision
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Data Processing**: NumPy, PIL
- **Evaluation**: scikit-learn, seaborn, matplotlib

---

## 📁 Dataset

[PlantVillage Dataset](https://www.kaggle.com/datasets/tairaislam/plantvillage-dataset) — Kaggle

---

## 👤 Author

**Vinamra** — [github.com/Vinamra31](https://github.com/Vinamra31)

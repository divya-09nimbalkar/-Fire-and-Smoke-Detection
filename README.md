
```markdown
# 🔥 Fire and Smoke Detection

## 📖 Overview
This project focuses on detecting **fire** and **smoke** in images using deep learning (PyTorch).  
It provides a complete pipeline for data preprocessing, model training, evaluation, and inference.

---

## 🎯 Objectives
- Define the business problem (early fire/smoke detection for safety).
- Build a machine learning solution.
- Evaluate results with metrics and saved models.

---

## 📂 Folder Structure
FIRE_AND_SMOKE_DETECTION/
│
├── data/
│   ├── raw/                # Original dataset (fire, smoke, normal)
│   └── processed/          # Preprocessed splits (train, val, test)
│
├── docs/                   # Documentation (guides, references)
├── models/
│   └── final/              # Saved trained models (model_best.pt)
│
├── notebooks/              # Jupyter notebooks for experiments
├── src/                    # Source code
│   ├── config.py           # Configuration settings
│   ├── data_loader.py      # Data pipeline
│   ├── inference.py        # Inference script
│   ├── main.py             # Entry point
│   ├── model.py            # Model architecture
│   ├── train.py            # Training loop
│   └── utils.py            # Helper functions
│
├── tests/                  # Unit tests
│   ├── test_data_loader.py
│   ├── test_inference.py
│   └── test_model.py
│
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt
└── temp.jpg                # Sample image

---

## ⚙️ Setup
Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Run
Run the main entry point:

```bash
python src/main.py
```

Or run specific modules:
- **Training:** `python src/train.py`
- **Inference:** `python src/inference.py --image temp.jpg`

---

## 🧪 Testing
Validate components with:

```bash
pytest tests/
```

---

## 📈 Results
- Best trained model: `models/final/model_best.pt`
- Metrics: accuracy, precision, recall, F1-score

---

## 📜 License
This project is licensed under the terms of the LICENSE file in the repository.

---

## 🙌 Acknowledgements
- PyTorch for the deep learning framework
- Open-source fire/smoke datasets
```

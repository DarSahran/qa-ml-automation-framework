# 🧲 QA ML Automation Framework

![CI](https://github.com/DarSahran/qa-ml-automation-framework/actions/workflows/ci.yml/badge.svg)

A CI/CD-integrated QA automation framework for validating machine learning pipelines — including data validation, model training reproducibility, and prediction testing. Built with PyTest, YAML configs, and GitHub Actions for seamless testing and deployment.

---

## 📌 Features

- ✅ YAML-configurable test setup
- 🧪 PyTest-based test suite for data, training, and predictions
- 📋 Accuracy threshold & reproducibility testing
- 📁 Modular code: `scripts/`, `tests/`, `configs/`
- 🔀 Fully integrated with GitHub Actions for CI automation
- 💾 Auto-generates synthetic datasets with Faker

---

## 💠 Tech Stack

- Python 3.10+
- PyTest
- scikit-learn
- Pandas
- YAML
- Faker
- GitHub Actions (CI)
- Joblib

---

## 📂 Project Structure

```bash
qa-ml-automation-framework/
├── configs/
│   └── test_config.yaml
├── data/
│   └── sample_data.csv
├── models/
│   └── model.pkl
├── scripts/
│   ├── train.py
│   ├── predict.py
│   └── generate_data.py
├── tests/
│   ├── test_data_validation.py
│   ├── test_model_training.py
│   └── test_predictions.py
├── .github/workflows/
│   └── ci.yml
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Getting Started

```bash
# Clone repo
git clone https://github.com/DarSahran/qa-ml-automation-framework.git
cd qa-ml-automation-framework

# Setup virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python scripts/generate_data.py

# Run tests
pytest tests/
```

---

## 📈 CI/CD Automation

This framework automatically runs the full test suite on every push and pull request via GitHub Actions.

---

## 👨‍💻 Author

**Sahran Altaf**  
GitHub: [@DarSahran](https://github.com/DarSahran)  
LinkedIn: [SahranAltaf](https://www.linkedin.com/in/SahranAltaf)

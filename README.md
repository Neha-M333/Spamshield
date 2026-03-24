# 🛡️ SpamShield — Email Spam Classifier

A machine learning project that classifies emails as **Spam** or **Ham (legitimate)** using two models — Multinomial Naive Bayes and Linear SVM — with a fully interactive **Streamlit web app**.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📸 Preview

| Classifier | EDA | Model Performance |
|---|---|---|
| Real-time spam detection | Word clouds & distributions | Confusion matrices & ROC curves |

---

## 📁 Project Structure

```
spamshield/
│
├── spam_app.py                    # Streamlit web application
├── spam_email_classifier.py       # Core ML training & evaluation script
├── spam.csv                       # Dataset (SMS Spam Collection)
│
├── outputs/
│   ├── chart1_distribution.png    # Spam vs Ham pie chart
│   ├── chart2_wordcloud.png       # Spam word cloud
│   ├── cm_Linear_SVM.png          # SVM confusion matrix
│   ├── cm_Multinomial_Naive_Bayes.png
│   ├── roc_Linear_SVM.png         # SVM ROC curve
│   └── roc_Multinomial_Naive_Bayes.png
│
├── requirements.txt               # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/spamshield.git
cd spamshield
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate — macOS/Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run spam_app.py
```

The app will open automatically at `http://localhost:8501`

### 5. (Optional) Run the Training Script Only

```bash
python spam_email_classifier.py
```

This trains both models, prints metrics to console, and saves all charts as `.png` files.

---

## 🧠 Models

| Model | Vectorizer | Test Accuracy | ROC-AUC |
|---|---|---|---|
| Multinomial Naive Bayes | CountVectorizer | ~98.2% | 0.98 |
| Linear SVM | TF-IDF | ~98.3% | 0.99 |

Both models are wrapped in **sklearn Pipelines** (vectorizer → classifier) to prevent data leakage.

---

## 🖥️ App Features

### 🔍 Classifier Page
- Paste any email or select a pre-loaded example
- Switch between Naive Bayes and SVM in the sidebar
- Get instant **SPAM 🚫** or **HAM ✅** prediction
- Run a **batch demo** on 4 pre-written test emails at once

### 📊 EDA Page
- Spam vs Ham donut chart
- Message length distribution by category
- Word cloud of most common spam keywords
- Interactive data table preview (adjustable row count)

### 📈 Model Performance Page
- Side-by-side metrics comparison table
- Per-model tabs with:
  - Accuracy, Precision, Recall, F1, ROC-AUC cards
  - Confusion matrix heatmap (test set)
  - ROC curve (train vs test AUC)

---

## 📊 Dataset

**SMS Spam Collection Dataset**
- Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) via GitHub mirror
- 5,574 SMS messages (after deduplication: ~5,169)
- Class distribution: **87.4% Ham**, **12.6% Spam**
- Columns: `Category` (spam/ham), `Message` (text)

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
wordcloud
streamlit
```

Install all via:
```bash
pip install -r requirements.txt
```

---

## 📈 Results Summary

### Linear SVM (Test Set)
- True Negatives (Ham correctly identified): **1,104**
- False Positives (Ham flagged as Spam): **3**
- False Negatives (Spam missed): **17**
- True Positives (Spam correctly caught): **169**

### Multinomial Naive Bayes (Test Set)
- True Negatives: **1,103**
- False Positives: **4**
- False Negatives: **17**
- True Positives: **169**

> Both models perform near-identically on unseen data. SVM has a slight edge in precision (fewer false alarms).

---

## 🔑 Key Spam Signals (from Word Cloud)

Words most strongly associated with spam: `free`, `call`, `txt`, `claim`, `prize`, `urgent`, `mobile`, `won`, `reply`, `stop`, `cash`, `offer`

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

Made with ❤️ using Python, scikit-learn, and Streamlit.

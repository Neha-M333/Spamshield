# =============================================================================
# Email Spam Classifier — Naive Bayes & SVM
# =============================================================================

# SETUP (run once in terminal):
#   pip install pandas scikit-learn matplotlib seaborn wordcloud tabulate

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud, STOPWORDS

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report
)

# =============================================================================
# 1. LOAD & CLEAN DATA
# =============================================================================
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

df = pd.read_csv(
    "https://raw.githubusercontent.com/Apaulgithub/oibsip_taskno4/main/spam.csv",
    encoding='ISO-8859-1'
)

# Rename columns
df.rename(columns={"v1": "Category", "v2": "Message"}, inplace=True)

# Drop extra unnamed columns
df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Create binary label: 1 = spam, 0 = ham
df['Spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

print(f"Dataset shape after cleaning: {df.shape}")
print(f"Spam: {df['Spam'].sum()} | Ham: {(df['Spam']==0).sum()}")
print()

# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================
print("=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)

# --- Chart 1: Spam vs Ham distribution ---
spread = df['Category'].value_counts()
plt.figure(figsize=(5, 5))
spread.plot(kind='pie', autopct='%1.2f%%', cmap='Set1')
plt.title('Distribution of Spam vs Ham')
plt.tight_layout()
plt.savefig("chart1_distribution.png", dpi=150)
plt.show()
print("Chart saved: chart1_distribution.png")

# --- Chart 2: Word Cloud for Spam ---
df_spam = df[df['Category'] == 'spam'].copy()
comment_words = ''
stopwords = set(STOPWORDS)

for val in df_spam.Message:
    tokens = str(val).lower().split()
    comment_words += " ".join(tokens) + " "

wordcloud = WordCloud(
    width=1000, height=500,
    background_color='white',
    stopwords=stopwords,
    min_font_size=10,
    max_words=1000,
    colormap='gist_heat_r'
).generate(comment_words)

plt.figure(figsize=(6, 6))
plt.title('Most Used Words in Spam Messages', fontsize=15, pad=20)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("chart2_wordcloud.png", dpi=150)
plt.show()
print("Chart saved: chart2_wordcloud.png")

# =============================================================================
# 3. TRAIN / TEST SPLIT
# =============================================================================
print()
print("=" * 60)
print("STEP 3: Train / Test Split (75% / 25%)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    df.Message, df.Spam, test_size=0.25, random_state=42
)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# =============================================================================
# 4. HELPER: EVALUATE MODEL
# =============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """Fit model, print metrics, plot ROC curve and confusion matrices."""
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    # ROC-AUC (use decision_function if predict_proba not available)
    if hasattr(model, "predict_proba"):
        prob_train = model.predict_proba(X_train)[:, 1]
        prob_test  = model.predict_proba(X_test)[:, 1]
    else:
        prob_train = model.decision_function(X_train)
        prob_test  = model.decision_function(X_test)

    roc_train = roc_auc_score(y_train, prob_train)
    roc_test  = roc_auc_score(y_test,  prob_test)

    print(f"\n[{model_name}]")
    print(f"  Train ROC-AUC : {roc_train:.4f}")
    print(f"  Test  ROC-AUC : {roc_test:.4f}")

    # ROC curve
    fpr_tr, tpr_tr, _ = roc_curve(y_train, prob_train)
    fpr_te, tpr_te, _ = roc_curve(y_test,  prob_test)

    plt.figure(figsize=(5, 5))
    plt.plot([0,1],[0,1],'k--')
    plt.plot(fpr_tr, tpr_tr, label=f"Train AUC = {roc_train:.2f}")
    plt.plot(fpr_te, tpr_te, label=f"Test  AUC = {roc_test:.2f}")
    plt.legend()
    plt.title(f"ROC Curve — {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    fname = f"roc_{model_name.replace(' ','_')}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"  ROC chart saved: {fname}")

    # Confusion matrices
    cm_tr = confusion_matrix(y_train, y_pred_train)
    cm_te = confusion_matrix(y_test,  y_pred_test)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for cm, axis, title in [(cm_tr, ax[0], "Train"), (cm_te, ax[1], "Test")]:
        sns.heatmap(cm, annot=True, fmt='.4g', cmap="Oranges",
                    xticklabels=['Ham','Spam'],
                    yticklabels=['Ham','Spam'], ax=axis)
        axis.set_xlabel("Predicted"); axis.set_ylabel("True")
        axis.set_title(f"{title} Confusion Matrix — {model_name}")
    plt.tight_layout()
    fname = f"cm_{model_name.replace(' ','_')}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"  Confusion matrix saved: {fname}")

    # Classification reports
    print("\n  Train Classification Report:")
    print(classification_report(y_train, y_pred_train, target_names=['Ham','Spam']))

    print("\n  Test Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['Ham','Spam']))

    return {
        "model"         : model_name,
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "test_accuracy" : accuracy_score(y_test,  y_pred_test),
        "train_roc_auc" : roc_train,
        "test_roc_auc"  : roc_test,
        "train_f1"      : f1_score(y_train, y_pred_train, average='weighted'),
        "test_f1"       : f1_score(y_test,  y_pred_test,  average='weighted'),
    }

# =============================================================================
# 5A. MODEL 1 — MULTINOMIAL NAIVE BAYES
# =============================================================================
print()
print("=" * 60)
print("STEP 5A: Model 1 — Multinomial Naive Bayes")
print("=" * 60)

nb_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

nb_scores = evaluate_model(nb_pipeline, X_train, X_test, y_train, y_test,
                            model_name="Multinomial Naive Bayes")

# =============================================================================
# 5B. MODEL 2 — LINEAR SVM  (TF-IDF features for better SVM performance)
# =============================================================================
print()
print("=" * 60)
print("STEP 5B: Model 2 — Linear SVM (TF-IDF)")
print("=" * 60)

svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm',   LinearSVC(max_iter=2000))
])

svm_scores = evaluate_model(svm_pipeline, X_train, X_test, y_train, y_test,
                             model_name="Linear SVM")

# =============================================================================
# 6. COMPARISON TABLE
# =============================================================================
print()
print("=" * 60)
print("STEP 6: Model Comparison")
print("=" * 60)

results = pd.DataFrame([nb_scores, svm_scores])
results = results.set_index("model")
print(results.to_string())

# =============================================================================
# 7. SPAM DETECTOR FUNCTION (uses best model — SVM)
# =============================================================================
print()
print("=" * 60)
print("STEP 7: Spam Detector Demo")
print("=" * 60)

def detect_spam(email_text, model=svm_pipeline):
    """Return 'Spam' or 'Ham' for any email string."""
    prediction = model.predict([email_text])[0]
    return "🚫 SPAM Email!" if prediction == 1 else "✅ Ham (legitimate) Email"

test_emails = [
    "Free Tickets for IPL — click now to claim!",
    "Hey, are we still on for lunch tomorrow?",
    "WINNER! You have won a £1000 prize. Call NOW to claim.",
    "Can you please review the attached report before the meeting?",
]

for email in test_emails:
    print(f'  "{email[:60]}..."' if len(email) > 60 else f'  "{email}"')
    print(f"   → {detect_spam(email)}\n")

print("Done! All charts saved in the current directory.")
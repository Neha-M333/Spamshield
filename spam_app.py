"""
Spam Email Classifier — Streamlit Web App
Run: streamlit run spam_app.py
"""

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import io, base64

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve
)

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpamShield — Email Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

:root {
    --bg:        #0d0f14;
    --surface:   #161b24;
    --border:    #252d3d;
    --accent:    #e8ff57;
    --accent2:   #ff5f57;
    --text:      #e8ecf0;
    --muted:     #7a8499;
    --ham:       #4ade80;
    --spam:      #f87171;
    --radius:    12px;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.stButton > button {
    background: var(--accent) !important;
    color: #0d0f14 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.6rem !important;
    letter-spacing: 0.05em !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(232,255,87,0.25) !important;
}

.result-spam {
    background: linear-gradient(135deg, #2d1515 0%, #1a0a0a 100%);
    border: 2px solid var(--spam);
    border-radius: var(--radius);
    padding: 1.5rem 2rem;
    text-align: center;
    animation: pulse-red 2s infinite;
}
.result-ham {
    background: linear-gradient(135deg, #0d2d18 0%, #0a1a0e 100%);
    border: 2px solid var(--ham);
    border-radius: var(--radius);
    padding: 1.5rem 2rem;
    text-align: center;
    animation: pulse-green 2s infinite;
}
@keyframes pulse-red  { 0%,100%{box-shadow:0 0 0 0 rgba(248,113,113,.4)} 50%{box-shadow:0 0 16px 4px rgba(248,113,113,.15)} }
@keyframes pulse-green{ 0%,100%{box-shadow:0 0 0 0 rgba(74,222,128,.4)} 50%{box-shadow:0 0 16px 4px rgba(74,222,128,.15)} }

.result-label { font-family:'Space Mono',monospace; font-size:1.8rem; font-weight:700; }
.result-sub   { color:var(--muted); font-size:0.9rem; margin-top:0.4rem; }

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-val { font-family:'Space Mono',monospace; font-size:1.5rem; color:var(--accent); font-weight:700; }
.metric-lbl { color:var(--muted); font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; margin-top:0.2rem; }

[data-testid="stTextArea"] textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

div[data-testid="stDataFrame"] { border-radius: var(--radius); overflow: hidden; }

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--accent);
    margin-bottom: 0.8rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Data + model loading (cached) ──────────────────────────────────────────
@st.cache_data(show_spinner="Loading & cleaning dataset…")
def load_data():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/Apaulgithub/oibsip_taskno4/main/spam.csv",
        encoding='ISO-8859-1'
    )
    df.rename(columns={"v1": "Category", "v2": "Message"}, inplace=True)
    df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True)
    df.drop_duplicates(inplace=True)
    df['Spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
    return df

@st.cache_resource(show_spinner="Training models…")
def train_models(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df.Message, df.Spam, test_size=0.25, random_state=42
    )

    nb = Pipeline([('vec', CountVectorizer()), ('nb', MultinomialNB())])
    svm = Pipeline([('tfidf', TfidfVectorizer()), ('svm', LinearSVC(max_iter=2000))])

    nb.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    results = {}
    for name, model in [("Multinomial Naive Bayes", nb), ("Linear SVM", svm)]:
        yp = model.predict(X_test)
        yt = model.predict(X_train)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
            prob_tr = model.predict_proba(X_train)[:, 1]
        else:
            prob = model.decision_function(X_test)
            prob_tr = model.decision_function(X_train)

        results[name] = {
            "model": model,
            "accuracy":  accuracy_score(y_test, yp),
            "precision": precision_score(y_test, yp),
            "recall":    recall_score(y_test, yp),
            "f1":        f1_score(y_test, yp),
            "roc_auc":   roc_auc_score(y_test, prob),
            "cm":        confusion_matrix(y_test, yp),
            "roc_data":  roc_curve(y_test, prob),
            "roc_data_tr": roc_curve(y_train, prob_tr),
            "roc_tr":    roc_auc_score(y_train, prob_tr),
        }

    return results, X_test, y_test

def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#161b24", dpi=130)
    buf.seek(0)
    return buf

# ─── Load ───────────────────────────────────────────────────────────────────
df = load_data()
model_results, X_test, y_test = train_models(df)

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ SpamShield")
    st.markdown("<div class='section-title'>Dataset Overview</div>", unsafe_allow_html=True)
    total   = len(df)
    n_spam  = df['Spam'].sum()
    n_ham   = total - n_spam
    st.markdown(f"""
    <div class='metric-card' style='margin-bottom:0.6rem'>
        <div class='metric-val'>{total:,}</div>
        <div class='metric-lbl'>Total Messages</div>
    </div>
    <div style='display:flex;gap:0.5rem;margin-bottom:1.2rem'>
        <div class='metric-card' style='flex:1'>
            <div class='metric-val' style='color:#4ade80'>{n_ham:,}</div>
            <div class='metric-lbl'>Ham</div>
        </div>
        <div class='metric-card' style='flex:1'>
            <div class='metric-val' style='color:#f87171'>{n_spam:,}</div>
            <div class='metric-lbl'>Spam</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Active Model</div>", unsafe_allow_html=True)
    chosen_model = st.selectbox("", list(model_results.keys()), label_visibility="collapsed")

    st.markdown("<div class='section-title'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio("", ["🔍 Classifier", "📊 EDA", "📈 Model Performance"], label_visibility="collapsed")

# ─── Page: Classifier ───────────────────────────────────────────────────────
if page == "🔍 Classifier":
    st.markdown("# 🛡️ SpamShield")
    st.markdown("<p style='color:var(--muted);margin-top:-0.5rem;margin-bottom:2rem'>Real-time email spam detection powered by ML</p>", unsafe_allow_html=True)

    examples = {
        "✉️ Paste your own": "",
        "🚫 Prize winner scam": "WINNER!! You have won a £1000 prize. Call NOW to claim your reward before it expires!",
        "🚫 Free offer spam":   "Free Tickets for IPL — click now to claim! Limited time offer, don't miss out!",
        "✅ Normal work email": "Can you please review the attached report before our meeting tomorrow afternoon?",
        "✅ Casual message":    "Hey, are we still on for lunch tomorrow? Let me know if the timing works for you.",
    }

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_ex = st.selectbox("Load an example or paste your own:", list(examples.keys()))
        default_txt = examples[selected_ex]
        user_input = st.text_area("Email / message text:", value=default_txt, height=160,
                                  placeholder="Type or paste an email here…")

    with col2:
        st.markdown("<div style='height:2.3rem'></div>", unsafe_allow_html=True)
        if st.button("🔍  Analyse Message", use_container_width=True):
            if user_input.strip():
                model = model_results[chosen_model]["model"]
                pred  = model.predict([user_input])[0]

                if pred == 1:
                    st.markdown("""
                    <div class='result-spam'>
                        <div class='result-label' style='color:#f87171'>🚫 SPAM</div>
                        <div class='result-sub'>This message shows signs of being unsolicited spam.</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='result-ham'>
                        <div class='result-label' style='color:#4ade80'>✅ HAM</div>
                        <div class='result-sub'>This message appears to be legitimate.</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.warning("Please enter some text first.")

        # Quick batch test
        st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
        if st.button("⚡  Run Batch Demo", use_container_width=True):
            demos = [
                "WINNER! You have been selected for a £500 cash prize. Call now!",
                "Hey, can we reschedule our 3pm call to tomorrow?",
                "FREE entry to win FA Cup final tickets! Text GOAL to 87239",
                "Please find the updated invoice attached. Let me know if you have questions.",
            ]
            model = model_results[chosen_model]["model"]
            st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
            for msg in demos:
                pred = model.predict([msg])[0]
                icon  = "🚫" if pred == 1 else "✅"
                label = "SPAM" if pred == 1 else "HAM"
                color = "#f87171" if pred == 1 else "#4ade80"
                st.markdown(f"""
                <div style='background:var(--surface);border:1px solid var(--border);
                     border-radius:8px;padding:0.7rem 1rem;margin-bottom:0.5rem;
                     display:flex;align-items:center;gap:0.8rem'>
                    <span style='font-family:Space Mono,monospace;color:{color};
                          font-weight:700;min-width:4rem'>{icon} {label}</span>
                    <span style='color:var(--muted);font-size:0.88rem'>{msg[:80]}{"…" if len(msg)>80 else ""}</span>
                </div>""", unsafe_allow_html=True)

# ─── Page: EDA ──────────────────────────────────────────────────────────────
elif page == "📊 EDA":
    st.markdown("# Exploratory Data Analysis")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='section-title'>Class Distribution</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor="#161b24")
        vals   = [n_ham, n_spam]
        labels = [f"Ham\n{n_ham:,}", f"Spam\n{n_spam:,}"]
        colors = ["#4ade80", "#f87171"]
        wedges, texts, autotexts = ax.pie(
            vals, labels=labels, autopct='%1.1f%%', colors=colors,
            startangle=90, wedgeprops=dict(width=0.55),
            textprops=dict(color="white", fontsize=10)
        )
        for at in autotexts:
            at.set_fontsize(9); at.set_color("#0d0f14"); at.set_fontweight("bold")
        ax.set_facecolor("#161b24")
        ax.set_title("Spam vs Ham", color="white", fontsize=12, pad=12)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c2:
        st.markdown("<div class='section-title'>Message Length Distribution</div>", unsafe_allow_html=True)
        df['length'] = df['Message'].str.len()
        fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor="#161b24")
        ax.set_facecolor("#161b24")
        for cat, col in [("ham", "#4ade80"), ("spam", "#f87171")]:
            ax.hist(df[df['Category'] == cat]['length'], bins=40,
                    alpha=0.7, color=col, label=cat.capitalize(), edgecolor='none')
        ax.set_xlabel("Message length (chars)", color="white", fontsize=9)
        ax.set_ylabel("Count", color="white", fontsize=9)
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color("#252d3d")
        ax.legend(facecolor="#161b24", labelcolor="white", fontsize=9)
        ax.set_title("Message Length by Category", color="white", fontsize=12, pad=12)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("<div class='section-title' style='margin-top:1.5rem'>Top Spam Keywords — Word Cloud</div>", unsafe_allow_html=True)
    spam_text = " ".join(df[df['Category']=='spam']['Message'].str.lower())
    wc = WordCloud(width=1100, height=420, background_color="#161b24",
                   stopwords=set(STOPWORDS), colormap='YlOrRd',
                   max_words=120, min_font_size=10).generate(spam_text)
    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor="#161b24")
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_facecolor("#161b24")
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("<div class='section-title' style='margin-top:1.5rem'>Sample Messages</div>", unsafe_allow_html=True)
    n = st.slider("Rows to preview", 5, 30, 10)
    sample = df[['Category','Message']].sample(n, random_state=1).reset_index(drop=True)
    st.dataframe(sample, use_container_width=True)

# ─── Page: Model Performance ────────────────────────────────────────────────
elif page == "📈 Model Performance":
    st.markdown("# Model Performance")

    # Summary table
    st.markdown("<div class='section-title'>Comparison Table</div>", unsafe_allow_html=True)
    rows = []
    for name, r in model_results.items():
        rows.append({
            "Model":     name,
            "Accuracy":  f"{r['accuracy']:.4f}",
            "Precision": f"{r['precision']:.4f}",
            "Recall":    f"{r['recall']:.4f}",
            "F1 Score":  f"{r['f1']:.4f}",
            "ROC-AUC":   f"{r['roc_auc']:.4f}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    # Per-model drill-down
    st.markdown("<div class='section-title' style='margin-top:1.5rem'>Detailed View</div>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(list(model_results.keys()))

    for tab, (mname, r) in zip([tab1, tab2], model_results.items()):
        with tab:
            # Metric cards
            metrics = [
                ("Accuracy",  r['accuracy']),
                ("Precision", r['precision']),
                ("Recall",    r['recall']),
                ("F1 Score",  r['f1']),
                ("ROC-AUC",   r['roc_auc']),
            ]
            cols = st.columns(5)
            for col, (lbl, val) in zip(cols, metrics):
                col.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-val'>{val:.3f}</div>
                    <div class='metric-lbl'>{lbl}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)

            # Confusion matrix
            with c1:
                cm = r['cm']
                fig, ax = plt.subplots(figsize=(4.5, 3.8), facecolor="#161b24")
                ax.set_facecolor("#161b24")
                sns.heatmap(cm, annot=True, fmt='d', cmap="YlOrRd",
                            xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'],
                            ax=ax, linewidths=0.5, linecolor="#161b24",
                            annot_kws={"fontsize": 13, "fontweight": "bold"})
                ax.set_xlabel("Predicted", color="white"); ax.set_ylabel("True", color="white")
                ax.tick_params(colors='white')
                ax.set_title(f"Confusion Matrix — Test Set", color="white", fontsize=11, pad=10)
                fig.patch.set_facecolor("#161b24")
                st.pyplot(fig, use_container_width=True)
                plt.close()

            # ROC curve
            with c2:
                fpr_te, tpr_te, _ = r['roc_data']
                fpr_tr, tpr_tr, _ = r['roc_data_tr']
                fig, ax = plt.subplots(figsize=(4.5, 3.8), facecolor="#161b24")
                ax.set_facecolor("#161b24")
                ax.plot([0,1],[0,1],'--', color="#7a8499", linewidth=1)
                ax.plot(fpr_tr, tpr_tr, color="#e8ff57", label=f"Train AUC = {r['roc_tr']:.2f}", lw=2)
                ax.plot(fpr_te, tpr_te, color="#f87171", label=f"Test  AUC = {r['roc_auc']:.2f}", lw=2)
                ax.legend(facecolor="#161b24", labelcolor="white", fontsize=9)
                ax.set_xlabel("False Positive Rate", color="white"); ax.set_ylabel("True Positive Rate", color="white")
                ax.tick_params(colors='white')
                for spine in ax.spines.values(): spine.set_color("#252d3d")
                ax.set_title("ROC Curve", color="white", fontsize=11, pad=10)
                fig.patch.set_facecolor("#161b24")
                st.pyplot(fig, use_container_width=True)
                plt.close()

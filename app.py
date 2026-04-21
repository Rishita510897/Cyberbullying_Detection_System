# =====================================================
# CyberShield — Cyberbullying Detection App
# Features: Toxicity Sub-categories, Multi-language,
#   Word Cloud, Sentiment Trend, Safe Message Suggestion,
#   PDF Report Export
# Run: streamlit run app.py
# =====================================================

import streamlit as st
import joblib
import re
import nltk
import numpy as np
import pandas as pd
import time
import io
import base64
from datetime import datetime
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from fpdf import FPDF
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# googletrans — graceful fallback if unavailable
try:
    from googletrans import Translator
    _translator = Translator()
    TRANSLATE_OK = True
except Exception:
    TRANSLATE_OK = False

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CyberShield — Bullying Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background: #f5f6f8;
    color: #1a1d23;
}
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e4e9;
}
[data-testid="stSidebar"] * { color: #1a1d23 !important; }
.main .block-container { padding: 2rem 2.5rem; max-width: 1200px; }
h1,h2,h3 { font-family:'IBM Plex Sans',sans-serif !important; font-weight:700 !important; color:#1a1d23 !important; }

textarea, input[type="text"] {
    background:#ffffff !important; border:1px solid #d0d3da !important;
    border-radius:6px !important; font-family:'IBM Plex Sans',sans-serif !important;
    font-size:0.9rem !important; color:#1a1d23 !important;
}
textarea:focus, input[type="text"]:focus {
    border-color:#2563eb !important; box-shadow:0 0 0 3px rgba(37,99,235,0.12) !important;
}
[data-testid="stButton"] > button {
    background:#2563eb !important; color:#ffffff !important; border:none !important;
    border-radius:6px !important; font-weight:600 !important;
    font-family:'IBM Plex Sans',sans-serif !important; padding:0.5rem 1.4rem !important;
}
[data-testid="stButton"] > button:hover { background:#1d4ed8 !important; }
[data-testid="stMetric"] {
    background:#ffffff !important; border:1px solid #e2e4e9 !important;
    border-radius:8px !important; padding:0.9rem 1rem !important;
}
[data-testid="stMetricLabel"] p { font-size:0.72rem !important; color:#6b7280 !important; text-transform:uppercase; letter-spacing:0.06em; }
[data-testid="stMetricValue"]   { font-size:1.3rem !important; color:#1a1d23 !important; font-weight:700 !important; }
[data-testid="stTabs"] [data-baseweb="tab"] { font-family:'IBM Plex Sans',sans-serif; font-weight:500; font-size:0.88rem; color:#6b7280; }
[data-testid="stTabs"] [aria-selected="true"] { color:#2563eb !important; font-weight:600 !important; }
hr { border-color:#e2e4e9 !important; }

.vbox { border-radius:8px; padding:1rem 1.3rem; margin-bottom:1.2rem; display:flex; align-items:flex-start; gap:0.9rem; }
.vbox-danger { background:#fef2f2; border:1px solid #fca5a5; border-left:4px solid #ef4444; }
.vbox-safe   { background:#f0fdf4; border:1px solid #86efac; border-left:4px solid #22c55e; }
.vbox-icon   { font-size:1.5rem; line-height:1; margin-top:0.1rem; }
.vbox-title  { font-size:1rem; font-weight:700; margin-bottom:0.15rem; }
.vbox-sub    { font-size:0.8rem; color:#6b7280; }
.vbox-danger .vbox-title { color:#dc2626; }
.vbox-safe   .vbox-title { color:#16a34a; }

.sec-label { font-size:0.68rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#9ca3af; margin-bottom:0.6rem; }

.icard { background:#ffffff; border:1px solid #e2e4e9; border-radius:8px; padding:1rem 1.2rem; margin-bottom:0.8rem; font-size:0.85rem; line-height:1.7; color:#374151; }

.tag { display:inline-block; padding:0.22rem 0.65rem; border-radius:99px; font-size:0.72rem; font-weight:600; margin:0.18rem 0.2rem 0 0; border:1px solid; }
.tag-y  { background:#fee2e2; color:#dc2626; border-color:#fca5a5; }
.tag-n  { background:#f3f4f6; color:#6b7280; border-color:#d1d5db; }
.tag-b  { background:#eff6ff; color:#2563eb; border-color:#bfdbfe; }
.tag-o  { background:#fff7ed; color:#c2410c; border-color:#fed7aa; }
.tag-p  { background:#fdf4ff; color:#7c3aed; border-color:#e9d5ff; }

.mono-box { background:#f9fafb; border:1px solid #e2e4e9; border-radius:6px; padding:0.7rem 0.9rem; font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#6b7280; word-break:break-word; line-height:1.65; min-height:48px; }

.suggest-box { background:#eff6ff; border:1px solid #bfdbfe; border-left:4px solid #2563eb; border-radius:8px; padding:1rem 1.2rem; font-size:0.88rem; color:#1e40af; line-height:1.6; margin-top:0.5rem; }

.tox-row { display:flex; gap:0.5rem; flex-wrap:wrap; margin-top:0.4rem; }
.tox-pill { padding:0.3rem 0.75rem; border-radius:6px; font-size:0.75rem; font-weight:600; border:1px solid; }
.tox-high { background:#fef2f2; color:#dc2626; border-color:#fca5a5; }
.tox-mid  { background:#fff7ed; color:#c2410c; border-color:#fed7aa; }
.tox-low  { background:#f0fdf4; color:#15803d; border-color:#86efac; }

[data-testid="stProgress"] > div > div { background-color:#2563eb !important; border-radius:99px; }
[data-testid="stDataFrame"] { border:1px solid #e2e4e9; border-radius:8px; overflow:hidden; }

.stat-box { background:#ffffff; border:1px solid #e2e4e9; border-radius:8px; padding:1.1rem 1.3rem; text-align:center; }
.stat-num { font-size:2rem; font-weight:700; color:#1a1d23; line-height:1.1; }
.stat-lbl { font-size:0.75rem; color:#9ca3af; margin-top:0.2rem; text-transform:uppercase; letter-spacing:0.06em; }
.sb-brand { font-size:1.25rem; font-weight:700; color:#1a1d23; letter-spacing:-0.01em; }
.sb-brand span { color:#2563eb; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, default in [("total_predictions", 0), ("harmful_count", 0)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# NLTK
# ─────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ─────────────────────────────────────────────
# TOXICITY SUB-CATEGORIES
# ─────────────────────────────────────────────
TOXICITY_CATEGORIES = {
    "Hate Speech":      ["hate", "racist", "sexist", "bigot", "slur", "discriminate",
                         "inferior", "subhuman", "filth", "vermin"],
    "Threats":          ["kill", "murder", "die", "hurt", "attack", "destroy", "beat",
                         "shoot", "stab", "harm", "threat", "end you"],
    "Insults":          ["idiot", "stupid", "dumb", "moron", "loser", "worthless",
                         "pathetic", "ugly", "fat", "freak", "trash", "disgusting"],
    "Identity Attacks": ["gay", "lesbian", "trans", "retard", "autistic", "disabled",
                         "foreigner", "immigrant", "religion", "race", "gender"],
}

def detect_toxicity_categories(text: str) -> dict:
    """Return dict of category -> {detected, matches}."""
    t = text.lower()
    results = {}
    for cat, keywords in TOXICITY_CATEGORIES.items():
        matched = [w for w in keywords if w in t]
        results[cat] = {"detected": bool(matched), "matches": matched}
    return results

def toxicity_severity(cat_results: dict) -> str:
    hits = sum(1 for v in cat_results.values() if v["detected"])
    if hits >= 3: return "High"
    if hits == 2: return "Medium"
    if hits == 1: return "Low"
    return "None"

# ─────────────────────────────────────────────
# MULTI-LANGUAGE DETECTION & TRANSLATION
# ─────────────────────────────────────────────
def detect_and_translate(text: str):
    """Returns (translated_text, lang_code, lang_name)."""
    if not TRANSLATE_OK or not text.strip():
        return text, "en", "English"
    try:
        detected  = _translator.detect(text)
        lang_code = detected.lang if detected else "en"
        lang_names = {
            "hi": "Hindi", "te": "Telugu", "ta": "Tamil", "fr": "French",
            "de": "German", "es": "Spanish", "ar": "Arabic", "zh-cn": "Chinese",
            "ja": "Japanese", "ko": "Korean", "pt": "Portuguese", "ru": "Russian",
            "it": "Italian", "en": "English",
        }
        lang_name  = lang_names.get(lang_code, lang_code.upper())
        translated = _translator.translate(text, dest="en").text if lang_code != "en" else text
        return translated, lang_code, lang_name
    except Exception:
        return text, "en", "English"

# ─────────────────────────────────────────────
# SAFE MESSAGE SUGGESTION
# ─────────────────────────────────────────────
REPLACEMENT_MAP = {
    "idiot": "person", "stupid": "mistaken", "dumb": "confused",
    "loser": "individual", "hate": "strongly dislike", "kill": "stop",
    "die": "go away", "ugly": "different", "fat": "larger",
    "worthless": "struggling", "trash": "unhelpful", "pathetic": "frustrating",
    "freak": "unique person", "moron": "person", "useless": "unhelpful",
    "shut up": "please stop", "go die": "please leave",
    "nobody likes you": "you may feel lonely",
}

def suggest_safe_message(text: str) -> str:
    """Replace toxic words with neutral alternatives."""
    result = text
    for toxic, safe in REPLACEMENT_MAP.items():
        result = re.sub(re.escape(toxic), safe, result, flags=re.IGNORECASE)
    result = result.strip()
    if result:
        result = result[0].upper() + result[1:]
        if not result.endswith(('.', '!', '?')):
            result += '.'
    return result

# ─────────────────────────────────────────────
# NLP HELPERS
# ─────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

def get_sentiment(text: str) -> float:
    return TextBlob(text).sentiment.polarity

def sentiment_label(score: float) -> str:
    if score > 0.1:  return "Positive"
    if score < -0.1: return "Negative"
    return "Neutral"

def sentiment_emoji(score: float) -> str:
    if score > 0.1:  return "Positive 😊"
    if score < -0.1: return "Negative 😠"
    return "Neutral 😐"

SARCASM_PHRASES = ["yeah right", "sure", "nice job", "great",
                   "wow", "amazing", "brilliant", "well done"]

def detect_sarcasm(text: str) -> int:
    t = text.lower()
    if any(p in t for p in SARCASM_PHRASES): return 1
    if "!" in t and any(w in t for w in ["great", "nice", "amazing"]): return 1
    return 0

GENDER_WORDS   = ["she", "he", "woman", "man", "girl", "boy"]
ATTACK_PHRASES = ["you are", "go die", "idiot", "stupid", "hate you"]

def class_features(text: str):
    t = text.lower()
    return (int(any(w in t for w in GENDER_WORDS)),
            int(any(p in t for p in ATTACK_PHRASES)))

ANGER_WORDS = ["hate", "angry", "idiot", "stupid", "useless", "kill"]
SAD_WORDS   = ["sad", "cry", "depressed", "lonely"]
FEAR_WORDS  = ["scared", "afraid", "fear", "threat"]

def detect_emotion(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ANGER_WORDS): return "Anger"
    if any(w in t for w in SAD_WORDS):   return "Sadness"
    if any(w in t for w in FEAR_WORDS):  return "Fear"
    return "Neutral"

def detect_emotion_emoji(text: str) -> str:
    e = detect_emotion(text)
    return {"Anger": "😡 Anger", "Sadness": "😢 Sadness",
            "Fear": "😨 Fear", "Neutral": "😐 Neutral"}[e]

def generate_reason(sentiment: float, sarcasm: int, g_flag: int, a_flag: int) -> str:
    reasons = []
    if sentiment < 0: reasons.append("negative emotional tone")
    if sarcasm:       reasons.append("sarcastic language")
    if g_flag:        reasons.append("gender-related reference")
    if a_flag:        reasons.append("direct personal attack")
    if not reasons:   return "No harmful patterns were detected in this message."
    return "Flagged because it contains: " + ", ".join(reasons) + "."

def build_feature_vector(text: str, vectorizer):
    clean     = preprocess(text)
    X_text    = vectorizer.transform([clean]).toarray()
    sentiment = np.array([[get_sentiment(clean)]])
    sarcasm   = np.array([[detect_sarcasm(text)]])
    g, a      = class_features(text)
    extra     = np.array([[g, a]])
    return (np.hstack((X_text, sentiment, sarcasm, extra)),
            clean, float(sentiment[0][0]), int(sarcasm[0][0]), g, a)

# ─────────────────────────────────────────────
# WORD CLOUD
# ─────────────────────────────────────────────
def generate_wordcloud(texts: list, title: str, colormap: str = "Reds") -> plt.Figure:
    combined = " ".join([preprocess(t) for t in texts])
    if not combined.strip():
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                fontsize=14, color="#9ca3af")
        ax.axis("off")
        return fig
    wc  = WordCloud(width=700, height=300, background_color="white",
                    colormap=colormap, max_words=80,
                    stopwords=stop_words).generate(combined)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", color="#1a1d23", pad=8)
    fig.tight_layout(pad=0.5)
    return fig

# ─────────────────────────────────────────────
# SENTIMENT TREND CHART
# ─────────────────────────────────────────────
def plot_sentiment_trend(sentiments: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 3))
    colors  = ["#ef4444" if s < -0.1 else "#22c55e" if s > 0.1 else "#94a3b8"
               for s in sentiments]
    ax.bar(range(len(sentiments)), sentiments, color=colors, alpha=0.85, width=0.7)
    ax.axhline(0, color="#d1d5db", linewidth=1)
    ax.set_xlabel("Message Index", fontsize=9, color="#6b7280")
    ax.set_ylabel("Polarity",      fontsize=9, color="#6b7280")
    ax.set_title("Sentiment Polarity Across Messages", fontsize=11,
                 fontweight="bold", color="#1a1d23")
    ax.tick_params(colors="#6b7280", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e2e4e9")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────
# PDF SANITIZER
# ─────────────────────────────────────────────
def sanitize(text: str) -> str:
    """Strip/replace chars outside latin-1 to prevent FPDF UnicodeEncodeError."""
    table = {
        "\u2014": "-",  "\u2013": "-",  "\u2018": "'",  "\u2019": "'",
        "\u201c": '"',  "\u201d": '"',  "\u2022": "*",  "\u2026": "...",
        "\u00b7": ".",  "\u00d7": "x",  "\u2192": "->", "\u2713": "Yes",
        "\u2717": "No", "\u26a0": "(!)","\u2705": "[OK]","\u274c": "[X]",
        "\U0001f600": "", "\U0001f601": "", "\U0001f610": "",
        "\U0001f621": "", "\U0001f622": "", "\U0001f628": "",
        "\U0001f6a8": "", "\U0001f6e1": "",
    }
    for char, rep in table.items():
        text = text.replace(char, rep)
    return text.encode("latin-1", errors="ignore").decode("latin-1")

# ─────────────────────────────────────────────
# PDF REPORT GENERATOR
# ─────────────────────────────────────────────
def generate_pdf_report(data: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Header bar
    pdf.set_fill_color(37, 99, 235)
    pdf.rect(0, 0, 210, 18, "F")
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(255, 255, 255)
    pdf.set_y(4)
    pdf.cell(0, 10, "CyberShield - Cyberbullying Detection Report", align="C")
    pdf.set_y(24)
    pdf.set_text_color(30, 30, 30)

    # Timestamp
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%d %b %Y  %H:%M:%S')}", ln=True)
    pdf.ln(3)

    def section(title):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(37, 99, 235)
        pdf.cell(0, 8, sanitize(title), ln=True)
        pdf.set_draw_color(191, 219, 254)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(2)
        pdf.set_text_color(30, 30, 30)

    def row(label, value, color=None):
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(55, 7, sanitize(str(label) + ":"), ln=False)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*(color if color else (30, 30, 30)))
        pdf.cell(0, 7, sanitize(str(value)), ln=True)
        pdf.set_text_color(30, 30, 30)

    def multi(text):
        pdf.multi_cell(0, 6, sanitize(str(text)), border=0)

    # Input
    section("Input Message")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(55, 65, 81)
    multi(data.get("original_text", ""))
    pdf.ln(3)

    # Verdict
    section("Prediction Result")
    pred_color = (220, 38, 38) if data["prediction"] == 1 else (22, 163, 74)
    pred_text  = "CYBERBULLYING DETECTED" if data["prediction"] == 1 else "SAFE MESSAGE"
    row("Verdict",    pred_text,                color=pred_color)
    row("Confidence", f"{data['confidence']:.1f}%")
    row("Risk Score", f"{data['severity']}/100")
    pdf.ln(2)

    # Language
    section("Language Detection")
    row("Detected Language", data.get("lang_name", "English"))
    row("Translated",        "Yes" if data.get("was_translated") else "No")
    pdf.ln(2)

    # NLP Features
    section("NLP Feature Analysis")
    row("Sentiment Polarity", f"{data['sentiment']:+.3f}  ({sentiment_label(data['sentiment'])})")
    row("Emotion",            data.get("emotion", "Neutral"))
    row("Sarcasm Detected",   "Yes" if data["sarcasm"]       else "No")
    row("Gender Reference",   "Yes" if data["gender_ref"]    else "No")
    row("Direct Attack",      "Yes" if data["direct_attack"] else "No")
    pdf.ln(2)

    # Toxicity
    section("Toxicity Sub-Categories")
    for cat, result in data.get("toxicity", {}).items():
        status  = "Detected" if result["detected"] else "Not Detected"
        matches = ", ".join(result["matches"]) if result["matches"] else "none"
        color   = (220, 38, 38) if result["detected"] else (107, 114, 128)
        row(cat, f"{status}  |  Keywords: {matches}", color=color)
    pdf.ln(2)

    # Safe suggestion
    if data["prediction"] == 1:
        section("Safe Message Suggestion")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(30, 64, 175)
        multi(data.get("suggestion", "No suggestion available."))
        pdf.ln(2)

    # Explanation
    section("Explanation")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(55, 65, 81)
    multi(data.get("reason", "No explanation available."))

    # Footer
    pdf.set_y(-15)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(180, 180, 180)
    pdf.cell(0, 10, "CyberShield  Real-Time Cyberbullying Detection  NLP + SVM", align="C")

    # Compatible with old fpdf (str) and fpdf2 (bytearray)
    raw = pdf.output(dest="S")
    if isinstance(raw, str):
        return raw.encode("latin-1", errors="replace")
    return bytes(raw)

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        m = joblib.load("model_joblib.pkl")
        v = joblib.load("vectorizer_joblib.pkl")
        return m, v
    except Exception:
        return None, None

model, vectorizer = load_model()

# ─────────────────────────────────────────────
# TAG HELPER
# ─────────────────────────────────────────────
def tag(label, active):
    cls = "tag-y" if active else "tag-n"
    val = "Yes" if active else "No"
    return f'<span class="tag {cls}">{label} &middot; {val}</span>'

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-brand">Cyber<span>Shield</span></div>',
                unsafe_allow_html=True)
    st.caption("Real-Time Bullying Detection · NLP + SVM")
    st.divider()
    st.markdown("**About**")
    st.markdown(
        "Analyzes text using **TF-IDF** + **SVM** with explainability signals: "
        "sentiment, sarcasm, toxicity sub-categories, emotion, and more."
    )
    st.divider()
    st.markdown(
        '<div style="font-size:0.72rem;color:#9ca3af">CyberShield · Cyberbullying Detection</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────
st.markdown("## 🛡️ Cyberbullying Detection System")
st.caption("Explainable, emotion-aware, multi-language real-time detection using NLP & SVM.")
st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["✍️  Manual Analysis", "📂  Batch CSV Analysis"])


# ══════════════════════════════════════════════
# TAB 1 — MANUAL ANALYSIS
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec-label">📝 Message Input</div>', unsafe_allow_html=True)

    col_in, col_hint = st.columns([1.1, 0.9], gap="large")

    with col_in:
        prev2       = st.text_input("Message before previous *(optional)*",
                                    placeholder="e.g.  Hey, did you see that?")
        prev1       = st.text_input("Previous message *(optional)*",
                                    placeholder="e.g.  What did you do yesterday?")
        user_input  = st.text_area("Current message *(required)*", height=120,
                                   placeholder="e.g.  You're such a loser, nobody likes you!")
        analyze_btn = st.button("🔍  Analyze Message", use_container_width=True)

    with col_hint:
        st.markdown("""
        <div class="icard">
            <b>How it works</b><br>
            1. Message is auto-detected for language &amp; translated to English.<br>
            2. Text is preprocessed (lemmatization, stopword removal).<br>
            3. TF-IDF + NLP features are extracted.<br>
            4. SVM classifies the message as Bullying / Not Bullying.<br>
            5. Toxicity sub-categories, suggestions &amp; PDF report are generated.
        </div>
        <div class="icard" style="border-left:3px solid #2563eb">
            <b>💡 Tip</b><br>
            The <em>previous messages</em> fields are optional but help the
            model understand conversation context for better accuracy.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── ANALYSIS LOGIC ──────────────────────────
    if analyze_btn:
        if not user_input.strip():
            st.warning("Please enter a **Current Message** to analyze.", icon="⚠️")
        else:
            full_text = f"{prev2} {prev1} {user_input}".strip()

            # Multi-language translation
            translated_text, lang_code, lang_name = detect_and_translate(full_text)
            was_translated = (lang_code != "en")
            if was_translated:
                st.info(
                    f"🌐 Detected language: **{lang_name}**  →  Translated to English for analysis.",
                    icon="🌐",
                )
            analysis_text = translated_text

            if model:
                X_final, clean_text, sent_score, sarcasm_flag, g_flag, a_flag = \
                    build_feature_vector(analysis_text, vectorizer)
                t0         = time.time()
                prediction = int(model.predict(X_final)[0])
                prob       = float(model.predict_proba(X_final)[0][1])
                latency_ms = (time.time() - t0) * 1000
            else:
                clean_text     = preprocess(analysis_text)
                sent_score     = get_sentiment(clean_text)
                sarcasm_flag   = detect_sarcasm(analysis_text)
                g_flag, a_flag = class_features(analysis_text)
                prob           = 0.85 if a_flag else 0.12
                prediction     = 1 if a_flag else 0
                latency_ms     = 0.0

            severity     = int(prob * 100)
            emotion      = detect_emotion(analysis_text)
            emotion_emj  = detect_emotion_emoji(analysis_text)
            tox_results  = detect_toxicity_categories(analysis_text)
            tox_sev      = toxicity_severity(tox_results)
            reason       = generate_reason(sent_score, sarcasm_flag, g_flag, a_flag)
            suggestion   = suggest_safe_message(user_input) if prediction == 1 else ""

            st.session_state.total_predictions += 1
            if prediction == 1:
                st.session_state.harmful_count += 1

            # ── Verdict ──
            st.markdown('<div class="sec-label">📊 Result</div>', unsafe_allow_html=True)
            if prediction == 1:
                st.markdown("""
                <div class="vbox vbox-danger">
                    <div class="vbox-icon">⚠️</div>
                    <div>
                        <div class="vbox-title">Cyberbullying Detected</div>
                        <div class="vbox-sub">This message contains potentially harmful or abusive content.</div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="vbox vbox-safe">
                    <div class="vbox-icon">✅</div>
                    <div>
                        <div class="vbox-title">Safe Message</div>
                        <div class="vbox-sub">No cyberbullying patterns were detected in this message.</div>
                    </div>
                </div>""", unsafe_allow_html=True)

            # ── Metrics ──
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Risk Score",  f"{severity}/100")
            m2.metric("Confidence",  f"{prob*100:.1f}%")
            m3.metric("Sentiment",   f"{sent_score:+.2f}")
            m4.metric("Emotion",     emotion)
            m5.metric("Latency",     f"{latency_ms:.1f} ms" if model else "—")

            st.markdown('<div class="sec-label" style="margin-top:0.8rem">Risk Level</div>',
                        unsafe_allow_html=True)
            st.progress(severity / 100)
            st.divider()

            # ── Toxicity Sub-Categories ──
            st.markdown('<div class="sec-label">🧪 Toxicity Sub-Categories</div>',
                        unsafe_allow_html=True)
            tox_cols = st.columns(4)
            tox_clr  = {"High": "#dc2626", "Medium": "#c2410c",
                        "Low": "#15803d", "None": "#6b7280"}
            for i, (cat, res) in enumerate(tox_results.items()):
                with tox_cols[i]:
                    detected = res["detected"]
                    matches  = ", ".join(res["matches"]) if res["matches"] else "—"
                    pill_cls = "tox-high" if detected else "tox-low"
                    st.markdown(f"""
                    <div class="icard" style="text-align:center;padding:0.7rem 0.6rem">
                        <div style="font-size:0.7rem;font-weight:600;color:#9ca3af;
                                    text-transform:uppercase;margin-bottom:0.3rem">{cat}</div>
                        <div class="tox-pill {pill_cls}" style="display:inline-block">
                            {'Detected' if detected else 'Clean'}
                        </div>
                        <div style="font-size:0.7rem;color:#9ca3af;margin-top:0.35rem">
                            {matches[:30] + '...' if len(matches) > 30 else matches}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            tox_color = tox_clr.get(tox_sev, "#6b7280")
            st.markdown(
                f'<div style="font-size:0.8rem;color:{tox_color};font-weight:600;margin-top:0.2rem">'
                f'Overall Toxicity Severity: {tox_sev}</div>',
                unsafe_allow_html=True,
            )
            st.divider()

            # ── Explanation + Preprocessed + Suggestion ──
            col_a, col_b = st.columns(2, gap="large")

            with col_a:
                st.markdown('<div class="sec-label">🔬 Explanation</div>',
                            unsafe_allow_html=True)
                svm_cls = "tag-y" if prediction == 1 else "tag-n"
                svm_lbl = "Bullying" if prediction == 1 else "Safe"
                st.markdown(f"""
                {tag("Sarcasm",     sarcasm_flag)}
                {tag("Gender Ref.", g_flag)}
                {tag("Direct Atk", a_flag)}
                <span class="tag tag-b">Sentiment &middot; {sentiment_emoji(sent_score)}</span>
                <span class="tag tag-p">Emotion &middot; {emotion_emj}</span>
                <span class="tag {svm_cls}">SVM &middot; {svm_lbl}</span>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    f'<div class="icard" style="font-size:0.83rem">{reason}</div>',
                    unsafe_allow_html=True,
                )

                if prediction == 1:
                    st.markdown(
                        '<div class="sec-label" style="margin-top:0.6rem">'
                        '💬 Safe Message Suggestion</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="suggest-box">'
                        f'<b>Suggested rephrasing:</b><br>{suggestion}</div>',
                        unsafe_allow_html=True,
                    )

            with col_b:
                st.markdown('<div class="sec-label">⚙️ Preprocessed Text</div>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="mono-box">{clean_text}</div>',
                            unsafe_allow_html=True)

                pct     = int((sent_score + 1) / 2 * 100)
                bar_clr = "#22c55e" if sent_score >= 0 else "#ef4444"
                st.markdown(f"""
                <div style="margin-top:0.9rem;font-size:0.72rem;color:#9ca3af;
                             text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem">
                    Sentiment Polarity
                </div>
                <div style="font-size:0.9rem;font-weight:600;color:#1a1d23;margin-bottom:0.4rem">
                    {sentiment_emoji(sent_score)} &nbsp;({sent_score:+.3f})
                </div>
                <div style="height:6px;background:#e2e4e9;border-radius:99px;overflow:hidden">
                    <div style="width:{pct}%;height:100%;background:{bar_clr};border-radius:99px"></div>
                </div>
                """, unsafe_allow_html=True)

                if was_translated:
                    st.markdown(
                        f'<div style="margin-top:0.8rem;font-size:0.75rem;color:#6b7280">'
                        f'Original language: <b>{lang_name}</b></div>',
                        unsafe_allow_html=True,
                    )

            st.divider()

            # ── PDF REPORT EXPORT ──
            st.markdown('<div class="sec-label">📄 Export Report</div>',
                        unsafe_allow_html=True)
            report_data = {
                "original_text": user_input,
                "prediction":    prediction,
                "confidence":    prob * 100,
                "severity":      severity,
                "sentiment":     sent_score,
                "emotion":       emotion,
                "sarcasm":       sarcasm_flag,
                "gender_ref":    g_flag,
                "direct_attack": a_flag,
                "toxicity":      tox_results,
                "suggestion":    suggestion,
                "reason":        reason,
                "lang_name":     lang_name,
                "was_translated": was_translated,
            }
            pdf_bytes = generate_pdf_report(report_data)
            fname     = f"cybershield_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            st.download_button(
                label="⬇️  Download PDF Report",
                data=pdf_bytes,
                file_name=fname,
                mime="application/pdf",
            )

    elif not analyze_btn:
        st.markdown("""
        <div style="text-align:center;padding:2.5rem 0;color:#9ca3af">
            <div style="font-size:2.4rem;margin-bottom:0.5rem">🛡️</div>
            <div style="font-size:0.95rem;font-weight:600;color:#6b7280">
                Enter a message and click <em>Analyze</em>
            </div>
            <div style="font-size:0.8rem;margin-top:0.25rem">
                Verdict, toxicity breakdown, sentiment, safe suggestion &amp; PDF export will appear here.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — BATCH CSV ANALYSIS
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-label">📂 Upload CSV File</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="icard" style="font-size:0.82rem;margin-bottom:1rem">'
        'Upload a <b>CSV file</b> with a column named <code>text</code>. '
        'Results include prediction, confidence, severity, sentiment, emotion, '
        'toxicity category flags, word cloud and sentiment trend visualizations.'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        if model is None:
            st.error("Model not loaded. Place model_joblib.pkl & vectorizer_joblib.pkl "
                     "in the project root.", icon="⚠️")
        else:
            df_batch = pd.read_csv(uploaded_file)

            if 'text' not in df_batch.columns:
                st.error("CSV must contain a column named **text**.", icon="❌")
            else:
                with st.spinner("Running batch analysis…"):
                    # Preprocess
                    df_batch['clean_text'] = df_batch['text'].astype(str).apply(preprocess)

                    # Build feature matrix
                    X_text     = vectorizer.transform(df_batch['clean_text']).toarray()
                    sentiments = np.array(
                        [get_sentiment(t) for t in df_batch['clean_text']]
                    ).reshape(-1, 1)
                    sarcasms   = np.array(
                        [detect_sarcasm(t) for t in df_batch['text'].astype(str)]
                    ).reshape(-1, 1)
                    cls_flags  = np.array(
                        [list(class_features(t)) for t in df_batch['text'].astype(str)]
                    )
                    X_final    = np.hstack((X_text, sentiments, sarcasms, cls_flags))

                    t0      = time.time()
                    preds   = model.predict(X_final)                  # numpy int array
                    probs   = model.predict_proba(X_final)[:, 1]     # numpy float array
                    elapsed = (time.time() - t0) * 1000

                st.success(
                    f"Processed **{len(df_batch)}** rows in {elapsed:.1f} ms "
                    f"({elapsed / len(df_batch):.2f} ms/sample)",
                    icon="✅",
                )

                # ── Build result columns (all pure Python / pandas — no numpy str ops) ──
                df_batch['Prediction'] = [
                    "Bullying" if int(p) == 1 else "Safe" for p in preds
                ]
                df_batch['Confidence'] = [
                    f"{float(p)*100:.1f}%" for p in probs
                ]
                df_batch['Severity']   = [int(float(p) * 100) for p in probs]
                df_batch['Sentiment']  = [
                    round(get_sentiment(t), 3) for t in df_batch['clean_text']
                ]
                df_batch['Emotion']    = [
                    detect_emotion(t) for t in df_batch['text'].astype(str)
                ]

                # Toxicity flags
                for cat in TOXICITY_CATEGORIES:
                    df_batch[cat] = [
                        "Yes" if detect_toxicity_categories(str(t))[cat]["detected"] else "No"
                        for t in df_batch['text']
                    ]

                total    = len(df_batch)
                bullying = int(sum(1 for p in preds if int(p) == 1))
                safe_cnt = total - bullying
                rate     = (bullying / total * 100) if total > 0 else 0.0

                # ── Summary Stats ──
                s1, s2, s3, s4 = st.columns(4)
                s1.markdown(f'<div class="stat-box"><div class="stat-num">{total}</div>'
                            f'<div class="stat-lbl">Total Analyzed</div></div>',
                            unsafe_allow_html=True)
                s2.markdown(f'<div class="stat-box"><div class="stat-num" style="color:#dc2626">'
                            f'{bullying}</div><div class="stat-lbl">Bullying</div></div>',
                            unsafe_allow_html=True)
                s3.markdown(f'<div class="stat-box"><div class="stat-num" style="color:#16a34a">'
                            f'{safe_cnt}</div><div class="stat-lbl">Safe</div></div>',
                            unsafe_allow_html=True)
                s4.markdown(f'<div class="stat-box"><div class="stat-num">{rate:.1f}%</div>'
                            f'<div class="stat-lbl">Detection Rate</div></div>',
                            unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Sentiment Trend Chart ──
                st.markdown('<div class="sec-label">📈 Sentiment Trend</div>',
                            unsafe_allow_html=True)
                fig_trend = plot_sentiment_trend(df_batch['Sentiment'].tolist())
                st.pyplot(fig_trend, use_container_width=True)
                plt.close(fig_trend)

                st.divider()

                # ── Word Clouds ──
                st.markdown('<div class="sec-label">☁️ Word Clouds</div>',
                            unsafe_allow_html=True)
                wc1, wc2 = st.columns(2)

                bullying_texts = df_batch[df_batch['Prediction'] == "Bullying"]['text'].tolist()
                safe_texts     = df_batch[df_batch['Prediction'] == "Safe"]['text'].tolist()

                with wc1:
                    if bullying_texts:
                        fig_b = generate_wordcloud(
                            bullying_texts, "Harmful Messages - Top Words", "Reds")
                        st.pyplot(fig_b, use_container_width=True)
                        plt.close(fig_b)
                    else:
                        st.info("No bullying messages in this batch.")

                with wc2:
                    if safe_texts:
                        fig_s = generate_wordcloud(
                            safe_texts, "Safe Messages - Top Words", "Greens")
                        st.pyplot(fig_s, use_container_width=True)
                        plt.close(fig_s)
                    else:
                        st.info("No safe messages in this batch.")

                st.divider()

                # ── Results Table ──
                st.markdown('<div class="sec-label">📋 Results Table</div>',
                            unsafe_allow_html=True)
                display_cols = (
                    ['text', 'Prediction', 'Confidence', 'Severity', 'Sentiment', 'Emotion']
                    + list(TOXICITY_CATEGORIES.keys())
                )
                st.dataframe(
                    df_batch[display_cols],
                    use_container_width=True,
                    hide_index=True,
                )

                csv_out = df_batch[display_cols].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️  Download Results as CSV",
                    data=csv_out,
                    file_name="cyberbullying_results.csv",
                    mime="text/csv",
                )

                st.session_state.total_predictions += total
                st.session_state.harmful_count     += bullying


# ─────────────────────────────────────────────
# SESSION DASHBOARD
# ─────────────────────────────────────────────
st.divider()
st.markdown('<div class="sec-label">📈 Session Dashboard</div>', unsafe_allow_html=True)

total_s   = st.session_state.total_predictions
harmful_s = st.session_state.harmful_count
safe_s    = total_s - harmful_s
rate_s    = (harmful_s / total_s * 100) if total_s > 0 else 0.0

d1, d2, d3, d4 = st.columns(4)
d1.metric("Total Analyzed",    total_s)
d2.metric("Bullying Detected", harmful_s)
d3.metric("Safe Messages",     safe_s)
d4.metric("Detection Rate",    f"{rate_s:.1f}%")

if total_s > 0:
    st.progress(harmful_s / total_s, text=f"{rate_s:.1f}% harmful in this session")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="text-align:center;font-size:0.72rem;color:#9ca3af">'
    'CyberShield · Real-Time Cyberbullying Detection · NLP · TF-IDF · SVM · Streamlit'
    '</div>',
    unsafe_allow_html=True,
)
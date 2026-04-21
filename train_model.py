# =====================================================
# Cyberbullying Detection — Training Script
# App-Compatible Version
# Run: python train_model.py
# =====================================================

import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

print("🚀 Starting Training Pipeline...")

# ─────────────────────────────────────────────
# NLTK SETUP
# ─────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# ─────────────────────────────────────────────
# LOAD & CLEAN DATASET
# ─────────────────────────────────────────────
df = pd.read_csv("data.csv")

if 'text' not in df.columns or 'label' not in df.columns:
    raise ValueError("Dataset must contain 'text' and 'label' columns.")

df = df[['text', 'label']].dropna()

# Standardize labels (handle: "not bullying", "not_bullying", "not-bullying", etc.)
df['label'] = df['label'].astype(str).str.lower().str.strip()
df['label'] = df['label'].str.replace(r"[\s_-]+", " ", regex=True)
df['label'] = df['label'].apply(lambda x: 0 if "not" in x else 1).astype(int)

print("✅ Labels cleaned successfully")
print(df['label'].value_counts())
# TEXT PREPROCESSING
def preprocess(text: str) -> str:
    """Lowercase, remove non-alpha chars, lemmatize, remove stopwords."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)
df['clean_text'] = df['text'].astype(str).apply(preprocess)
# FEATURE ENGINEERING
# Sentiment polarity (TextBlob)
def get_sentiment(text: str) -> float:
    return TextBlob(text).sentiment.polarity

df['sentiment'] = df['clean_text'].apply(get_sentiment)

# Sarcasm detection (heuristic)
SARCASM_PHRASES = ["yeah right", "sure", "nice job", "great",
                   "wow", "amazing", "brilliant", "well done"]
def detect_sarcasm(text: str) -> int:
    text = text.lower()
    if any(p in text for p in SARCASM_PHRASES):
        return 1
    if "!" in text and any(w in text for w in ["great", "nice", "amazing"]):
        return 1
    return 0
df['sarcasm'] = df['text'].apply(detect_sarcasm)
# Gender & attack flags
GENDER_WORDS   = ["she", "he", "woman", "man", "girl", "boy"]
ATTACK_PHRASES = ["you are", "go die", "idiot", "stupid", "hate you"]
def class_features(text: str):
    """Return (gender_flag, attack_flag) as integers."""
    text = text.lower()
    has_gender = int(any(w in text for w in GENDER_WORDS))
    has_attack = int(any(p in text for p in ATTACK_PHRASES))
    return has_gender, has_attack

df[['gender_flag', 'attack_flag']] = df['text'].apply(
    lambda x: pd.Series(class_features(x))
)


# ─────────────────────────────────────────────
# TF-IDF VECTORIZATION
# ─────────────────────────────────────────────
tfidf    = TfidfVectorizer(max_features=3000)
X_text   = tfidf.fit_transform(df['clean_text']).toarray()
y        = df['label']


# ─────────────────────────────────────────────
# BASELINE MODEL  (TF-IDF only)
# ─────────────────────────────────────────────
print("\n📊 Training Baseline Model (TF-IDF only)...")

Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

baseline_model = SVC(kernel='linear')
baseline_model.fit(Xb_train, yb_train)

base_pred = baseline_model.predict(Xb_test)
base_acc  = accuracy_score(yb_test, base_pred)
print(f"Baseline Accuracy: {base_acc * 100:.2f}%")


# ─────────────────────────────────────────────
# PROPOSED HYBRID MODEL  (TF-IDF + NLP features)
# ─────────────────────────────────────────────
print("\n📊 Training Proposed Hybrid Model...")

X = np.hstack((
    X_text,
    df[['sentiment']].values,
    df[['sarcasm']].values,
    df[['gender_flag', 'attack_flag']].values
))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n📌 Classification Report (Proposed Model):")
print(classification_report(y_test, y_pred))
print(f"Proposed Model Accuracy : {accuracy * 100:.2f}%")
#print(f"Accuracy Improvement    : {(accuracy - base_acc) * 100:.2f}%")

# Confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix — Proposed Model")
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────
# SAVE MODEL & VECTORIZER
# ─────────────────────────────────────────────
joblib.dump(model, "model_joblib.pkl")
joblib.dump(tfidf, "vectorizer_joblib.pkl")

print("\n✅ Training Complete!")
print("   model_joblib.pkl      — saved")
print("   vectorizer_joblib.pkl — saved")

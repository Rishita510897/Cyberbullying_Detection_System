# =====================================================
# Cyberbullying Detection — Model Testing Script
# Run: python test_model.py
# =====================================================

import joblib
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

print("🚀 Loading Testing Pipeline...")

# ─────────────────────────────────────────────
# NLTK SETUP
# ─────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# ─────────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(text: str) -> str:
    """Lowercase, strip non-alpha, lemmatize, remove stopwords."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)


# ─────────────────────────────────────────────
# FEATURE FUNCTIONS  (must mirror train_model.py)
# ─────────────────────────────────────────────

def get_sentiment(text: str) -> float:
    """TextBlob polarity score in [-1, +1]."""
    return TextBlob(text).sentiment.polarity


SARCASM_PHRASES = ["yeah right", "sure", "nice job", "great",
                   "wow", "amazing", "brilliant", "well done"]

def detect_sarcasm(text: str) -> int:
    """Return 1 if sarcasm heuristic triggers."""
    text = text.lower()
    if any(p in text for p in SARCASM_PHRASES):
        return 1
    if "!" in text and any(w in text for w in ["great", "nice", "amazing"]):
        return 1
    return 0


GENDER_WORDS   = ["she", "he", "woman", "man", "girl", "boy"]
ATTACK_PHRASES = ["you are", "go die", "idiot", "stupid", "hate you"]

def class_features(text: str):
    """Return (gender_flag, attack_flag) as integers."""
    text = text.lower()
    has_gender = int(any(w in text for w in GENDER_WORDS))
    has_attack = int(any(p in text for p in ATTACK_PHRASES))
    return has_gender, has_attack


ANGER_WORDS = ["hate", "angry", "idiot", "stupid", "useless", "kill"]
SAD_WORDS   = ["sad", "cry", "depressed", "lonely"]
FEAR_WORDS  = ["scared", "afraid", "fear", "threat"]

def detect_emotion(text: str) -> str:
    """Simple keyword-based emotion classifier."""
    text = text.lower()
    if any(w in text for w in ANGER_WORDS):
        return "Anger"
    if any(w in text for w in SAD_WORDS):
        return "Sadness"
    if any(w in text for w in FEAR_WORDS):
        return "Fear"
    return "Neutral"


# ─────────────────────────────────────────────
# LOAD MODEL & VECTORIZER
# ─────────────────────────────────────────────
try:
    model      = joblib.load("model_joblib.pkl")
    vectorizer = joblib.load("vectorizer_joblib.pkl")
    print("✅ Model and vectorizer loaded successfully!\n")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()


# ─────────────────────────────────────────────
# SAMPLE PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict_sample(text: str) -> dict:
    """
    Run full prediction pipeline on a single text string.
    Returns a dict with all analysis fields.
    """
    clean      = preprocess(text)
    X_text     = vectorizer.transform([clean]).toarray()
    sentiment  = np.array([[get_sentiment(clean)]])
    sarcasm    = np.array([[detect_sarcasm(text)]])
    g_flag, a_flag = class_features(text)
    extra      = np.array([[g_flag, a_flag]])

    X_final    = np.hstack((X_text, sentiment, sarcasm, extra))
    pred       = model.predict(X_final)[0]
    prob       = model.predict_proba(X_final)[0][1]

    return {
        "text":       text,
        "prediction": "Bullying" if pred == 1 else "Not Bullying",
        "confidence": round(prob * 100, 2),
        "severity":   int(prob * 100),
        "emotion":    detect_emotion(text),
        "sentiment":  round(float(sentiment[0][0]), 3),
        "sarcasm":    int(sarcasm[0][0]),
        "gender_ref": g_flag,
        "attack":     a_flag,
    }


# ─────────────────────────────────────────────
# TEST SAMPLES
# ─────────────────────────────────────────────
SAMPLES = [
    "You are useless and stupid",
    "I hope you are doing well today",
    "Yeah great, nice job genius",
    "Go die idiot",
    "She is so kind and helpful",
    "Nobody likes you, just disappear",
]

print("▶  Running Sample Predictions...\n")
print("=" * 54)

for sample in SAMPLES:
    result = predict_sample(sample)

    label_icon = "🔴" if result["prediction"] == "Bullying" else "🟢"

    print(f"Text        : {result['text']}")
    print(f"Prediction  : {label_icon}  {result['prediction']}")
    print(f"Confidence  : {result['confidence']}%")
    print(f"Severity    : {result['severity']}/100")
    print(f"Emotion     : {result['emotion']}")
    print(f"Sentiment   : {result['sentiment']:+.3f}")
    print(f"Sarcasm     : {'Yes' if result['sarcasm']    else 'No'}")
    print(f"Gender Ref  : {'Yes' if result['gender_ref'] else 'No'}")
    print(f"Direct Atk  : {'Yes' if result['attack']     else 'No'}")
    print("-" * 54)

print("\n✅ Testing Completed Successfully.")

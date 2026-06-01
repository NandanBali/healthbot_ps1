"""
ML mood/emotion classifier (scikit-learn).

A TfidfVectorizer + LogisticRegression pipeline trained on a small in-module
labeled phrase dataset. Classifies short user messages into one of:
``positive``, ``calm``, ``anxious``, ``stressed``, ``sad``.

Used by the sidebar mood tracker and exposed to the chat agent as a tool.
This is a lightweight demo model — not a clinical sentiment instrument.
"""

import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

_ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
_ARTIFACT_PATH = os.path.join(_ARTIFACT_DIR, "mood_model.joblib")
_RANDOM_STATE = 42

_model: Pipeline | None = None

# Small labeled training set. Kept intentionally compact; each label has enough
# varied phrasings for TF-IDF + logistic regression to separate them.
_TRAINING_DATA: list[tuple[str, str]] = [
    # positive
    ("i feel great today", "positive"),
    ("i am so happy and grateful", "positive"),
    ("things are going really well for me", "positive"),
    ("i had a wonderful day", "positive"),
    ("i feel hopeful and excited about the future", "positive"),
    ("life is good and i am thriving", "positive"),
    ("i am proud of what i accomplished", "positive"),
    ("feeling joyful and energetic", "positive"),
    # calm
    ("i feel calm and relaxed", "calm"),
    ("everything is peaceful right now", "calm"),
    ("i am at ease and content", "calm"),
    ("i feel settled and steady today", "calm"),
    ("my mind is quiet and serene", "calm"),
    ("i am comfortable and balanced", "calm"),
    ("just a normal ordinary day", "calm"),
    ("i feel fine and stable", "calm"),
    # anxious
    ("i feel so anxious and worried", "anxious"),
    ("my heart is racing and i am nervous", "anxious"),
    ("i can't stop worrying about everything", "anxious"),
    ("i feel on edge and panicky", "anxious"),
    ("i am scared and full of dread", "anxious"),
    ("i feel restless and uneasy", "anxious"),
    ("i keep fearing something bad will happen", "anxious"),
    ("i am terrified and tense", "anxious"),
    # stressed
    ("i am so stressed and overwhelmed", "stressed"),
    ("there is too much pressure at work", "stressed"),
    ("i feel burned out and exhausted", "stressed"),
    ("i have way too much to handle right now", "stressed"),
    ("i am frustrated and overloaded", "stressed"),
    ("everything feels like too much to cope with", "stressed"),
    ("i am drained and stretched too thin", "stressed"),
    ("the deadlines are crushing me", "stressed"),
    # sad
    ("i feel so sad and down", "sad"),
    ("i am depressed and hopeless", "sad"),
    ("i feel empty and lonely", "sad"),
    ("nothing brings me joy anymore", "sad"),
    ("i have been crying and feel miserable", "sad"),
    ("i feel worthless and low", "sad"),
    ("i am heartbroken and gloomy", "sad"),
    ("i feel tired of everything and unhappy", "sad"),
]

_NEGATIVE_LABELS = {"anxious", "stressed", "sad"}


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=_RANDOM_STATE),
            ),
        ]
    )


def train(save: bool = True) -> Pipeline:
    """Train the mood model on the in-module dataset and optionally persist it."""
    texts = [t for t, _ in _TRAINING_DATA]
    labels = [lbl for _, lbl in _TRAINING_DATA]
    pipeline = _build_pipeline()
    pipeline.fit(texts, labels)
    if save:
        os.makedirs(_ARTIFACT_DIR, exist_ok=True)
        joblib.dump(pipeline, _ARTIFACT_PATH)
    return pipeline


def _get_model() -> Pipeline:
    global _model
    if _model is not None:
        return _model
    if os.path.exists(_ARTIFACT_PATH):
        _model = joblib.load(_ARTIFACT_PATH)
    else:
        _model = train(save=True)
    return _model


def analyze_mood(text: str) -> dict:
    """Classify ``text`` into a mood label with a confidence score.

    Returns a dict with keys:
      label  str   one of positive/calm/anxious/stressed/sad
      score  float confidence of the predicted label, in [0, 1]
    """
    if not text or not text.strip():
        return {"label": "calm", "score": 0.0}

    model = _get_model()
    proba = model.predict_proba([text])[0]
    classes = model.classes_
    idx = int(proba.argmax())
    return {"label": str(classes[idx]), "score": float(proba[idx])}


def analyze_mood_text(text: str) -> str:
    """String-returning wrapper for the chat function-calling tool."""
    result = analyze_mood(text)
    tone = "negative" if result["label"] in _NEGATIVE_LABELS else "non-negative"
    return (
        f"Detected mood: {result['label']} ({tone}), "
        f"confidence {result['score'] * 100:.0f}%. "
        f"(Demo scikit-learn model — not a clinical assessment.)"
    )

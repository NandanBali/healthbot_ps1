"""Machine-learning models for the cardiac/mental-health PoC.

Models are scikit-learn pipelines trained on programmatically generated
synthetic data (no network, no bundled data files). Artifacts are persisted
under ``app/ml/artifacts/`` with joblib and loaded lazily; if an artifact is
missing the module trains and saves it on first use.
"""

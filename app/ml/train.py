"""
CLI to (re)train both ML models and write their artifacts.

Usage:
    python -m app.ml.train
"""

from app.ml import cardiac_model, mood_model


def main() -> None:
    print("Training cardiac-risk model…")
    cardiac_model.train(save=True)
    print(f"  saved → {cardiac_model._ARTIFACT_PATH}")

    print("Training mood model…")
    mood_model.train(save=True)
    print(f"  saved → {mood_model._ARTIFACT_PATH}")

    print("Done.")


if __name__ == "__main__":
    main()

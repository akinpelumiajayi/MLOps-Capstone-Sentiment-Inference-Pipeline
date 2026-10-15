import joblib
import os
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

REVIEWS = [
    ("This model is surprisingly accurate and fast", 1),
    ("Absolutely love the predictions, very reliable", 1),
    ("Great performance, exceeded all expectations", 1),
    ("The API response time is excellent", 1),
    ("Impressive accuracy on edge cases", 1),
    ("Works perfectly, highly recommend it", 1),
    ("Solid results, very happy with the output", 1),
    ("Best classifier I have used in production", 1),
    ("Clean interface and accurate predictions", 1),
    ("Really useful tool, saves us hours every day", 1),
    ("Outstanding recall on negative samples", 1),
    ("Precision is top notch, minimal false positives", 1),
    ("Deployed smoothly, zero issues so far", 1),
    ("The confidence scores are well calibrated", 1),
    ("Handles noisy input better than expected", 1),
    ("Works great in production environments", 1),
    ("Very good results across all our test cases", 1),
    ("Accurate and fast, exactly what we needed", 1),
    ("The model generalises well to new data", 1),
    ("Reliable predictions every single time", 1),
    ("Performs well under high load conditions", 1),
    ("The inference latency is impressively low", 1),
    ("Love how easy it is to integrate", 1),
    ("Highly accurate on our real world dataset", 1),
    ("Excellent tool for sentiment classification tasks", 1),
    ("Terrible accuracy, completely unreliable results", 0),
    ("Way too many false positives, not production ready", 0),
    ("The model fails on basic examples consistently", 0),
    ("Predictions are random at best, totally broken", 0),
    ("Disappointed with the overall performance", 0),
    ("Very slow inference, unusable at scale", 0),
    ("Poor recall, misses obvious negative cases", 0),
    ("Results are inconsistent and unpredictable", 0),
    ("Not worth using in production, too many errors", 0),
    ("Crashes under load, very unstable deployment", 0),
    ("Confidence scores make absolutely no sense", 0),
    ("Awful precision, floods downstream with noise", 0),
    ("Fails on anything outside the training domain", 0),
    ("Latency spikes make it unreliable in production", 0),
    ("The output is broken and very hard to parse", 0),
    ("Completely wrong predictions on obvious inputs", 0),
    ("The model degraded badly after the first week", 0),
    ("Useless for our use case, poor generalisation", 0),
    ("Accuracy dropped below baseline after deployment", 0),
    ("Broken pipeline, cannot use this in production", 0),
    ("The classifier is biased and inconsistent", 0),
    ("Very poor results, we had to roll back immediately", 0),
    ("Does not work as advertised, very disappointing", 0),
    ("High error rate makes this unusable for us", 0),
    ("The model is not ready for any real workload", 0),
]


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            min_df=1,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )),
    ])


def train(output_path: str = "models/model.pkl") -> Pipeline:
    texts, labels = zip(*REVIEWS)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print(classification_report(
        y_test, pipeline.predict(X_test),
        target_names=["negative", "positive"]
    ))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"Model saved → {output_path}")
    return pipeline


def load(model_path: str = "models/model.pkl") -> Pipeline:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No model found at {model_path}. "
            "Run `python training/train.py` or `dvc repro` to train one first."
        )
    return joblib.load(path)


if __name__ == "__main__":
    train(output_path=os.getenv("MODEL_PATH", "models/model.pkl"))

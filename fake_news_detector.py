import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


def load_dataset(csv_path: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """Load a fake news dataset from CSV or return a small built-in sample."""
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must include 'text' and 'label' columns.")
        df = df.dropna(subset=['text', 'label'])
        return df['text'].astype(str).tolist(), df['label'].astype(str).tolist()

    sample_texts = [
        "Government sources say the economy is growing faster than expected.",
        "Scientists confirm that chocolate cures all types of diseases.",
        "A well-known news site reports new policy changes in the education sector.",
        "Celebrity claims a secret cure for weight loss and it goes viral online.",
        "Local community organizes a food drive after flood warnings.",
        "Fake report says aliens landed in the city park last night.",
        "Research team publishes peer-reviewed study on climate change impacts.",
        "Hoarders buy all the groceries because of a false emergency alert.",
    ]
    sample_labels = [
        'real',
        'fake',
        'real',
        'fake',
        'real',
        'fake',
        'real',
        'fake',
    ]
    return sample_texts, sample_labels


def build_pipeline(ngram_range: Tuple[int, int] = (1, 2), use_tfidf: bool = True) -> Pipeline:
    """Create a text classification pipeline with vectorization and logistic regression."""
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.85,
        ngram_range=ngram_range,
    )
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    return Pipeline([('vectorizer', vectorizer), ('classifier', classifier)])


def train_model(
    texts: List[str],
    labels: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, List[str], List[str], List[str], List[str]]:
    """Train the model and return the trained pipeline plus test data."""
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels if len(set(labels)) > 1 else None,
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline, X_train, X_test, y_train, y_test


def evaluate_model(model: Pipeline, X_test: List[str], y_test: List[str]) -> None:
    """Evaluate the trained model and print accuracy and classification report."""
    predictions = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, predictions))
    print('\nClassification report:')
    print(classification_report(y_test, predictions, zero_division=0))


def classify_text(model: Pipeline, texts: List[str]) -> List[str]:
    """Predict fake/real labels for the given texts."""
    return model.predict(texts).tolist()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fake news detector using NLP and ML.')
    parser.add_argument('--csv', type=str, help='Path to a CSV dataset with text and label columns.')
    parser.add_argument('--text', type=str, help='Single article text to classify.')
    parser.add_argument('--test-size', type=float, default=0.2, help='Fraction of dataset used for testing.')
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    print('Loading dataset...')
    texts, labels = load_dataset(args.csv)

    if len(texts) < 2:
        raise ValueError('Dataset must contain at least 2 samples.')

    print(f'Loaded {len(texts)} examples.')
    model, _, X_test, _, y_test = train_model(texts, labels, test_size=args.test_size)

    print('\nEvaluating model on the test split...')
    evaluate_model(model, X_test, y_test)

    if args.text:
        prediction = classify_text(model, [args.text])[0]
        print('\nInput text:')
        print(args.text)
        print('\nPredicted label:', prediction)
    else:
        print('\nNo input text provided for live prediction. Use --text "<article>" to classify one article.')


if __name__ == '__main__':
    main()

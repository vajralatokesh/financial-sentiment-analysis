import pandas as pd
import numpy as np
import time
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import json

def main():
    print("Loading Financial PhraseBank dataset...")
    # Load dataset
    with open("FinancialPhraseBank-v1.0/Sentences_50Agree.txt", "r", encoding="iso-8859-1") as f:
        lines = f.readlines()
    sentences, labels = [], []
    for line in lines:
        if '@' in line:
            parts = line.strip().rsplit('@', 1)
            sentences.append(parts[0])
            labels.append(parts[1])
    df = pd.DataFrame({'sentence': sentences, 'label': labels})
    
    # Map labels: 0: negative, 1: neutral, 2: positive
    # Our labels are currently strings: 'negative', 'neutral', 'positive'
    label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['label'] = df['label'].map(label_mapping)
    X = df['sentence']
    y = df['label']
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM (Linear)": SVC(kernel='linear', random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()
        model.fit(X_train_tfidf, y_train)
        train_time = time.time() - start_time
        
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results[name] = {
            "accuracy": acc,
            "train_time_sec": train_time,
            "classification_report": report
        }
        
        print(f"{name} Accuracy: {acc:.4f}")
        
    print("Saving results...")
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Done. Analysis complete.")

if __name__ == "__main__":
    main()

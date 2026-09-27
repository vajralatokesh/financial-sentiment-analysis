# Financial News Sentiment Analysis Using Machine Learning

This repository contains the implementation and evaluation of machine learning models for sentiment analysis on financial news texts.

## Dataset
The project uses the **Financial PhraseBank** dataset, a benchmark for financial sentiment analysis containing sentences from financial news. Sentences are classified as:
- Positive
- Neutral
- Negative

The `sentences_50agree` subset is used, ensuring that at least 50% of annotators agreed on the sentiment label.

## Literature Survey & Context
This project implements baselines based on methods discussed in the associated literature survey. A key finding in the literature is that traditional machine learning models (like SVM and Logistic Regression) provide a solid baseline for financial sentiment classification, while more modern models (like FinBERT or LLMs) require more computational overhead. Special attention is also given to the difficulty of neutral-class classification.

## Implementation
The feature extraction uses `TfidfVectorizer` (TF-IDF). Three classical machine learning classifiers are implemented and evaluated:
1. Logistic Regression
2. Support Vector Machine (Linear Kernel)
3. XGBoost

## Requirements
```bash
pip install -r requirements.txt
```

## Running the Code
Run the `analyze.py` script:
```bash
python analyze.py
```
This will download the dataset, train the models, evaluate their performance (accuracy, F1-score), and output the results to `results.json`.

## Results

The models achieved the following baseline accuracies using TF-IDF features on the `sentences_50agree` subset of the Financial PhraseBank:

- **Logistic Regression**: 75.36%
- **SVM (Linear)**: 76.49%
- **XGBoost**: 76.49%

These results align closely with the findings of the literature survey (e.g. Papers 6 and 7), demonstrating that classical machine learning approaches yield solid baselines for financial sentiment classification tasks even before resorting to heavier models like FinBERT.

## Next Steps
To further improve performance, particularly on neutral class, consider integrating a domain-specific lexicon or fine-tuning a transformer model like FinBERT.

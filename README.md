# Movie Review Sentiment Analysis

End-to-end sentiment analysis project using:

- TF-IDF feature extraction
- Logistic Regression classifier
- IMDb reviews dataset in CSV format
- Streamlit UI for interactive predictions

## Features

- Reusable text preprocessing pipeline
  - Lowercasing
  - Punctuation/special character removal
  - Tokenization
  - NLTK stopword removal
- TF-IDF vectorization with max 5000 features
- Logistic Regression model training with 80/20 train-test split
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
- Model persistence via pickle:
  - model.pkl
  - vectorizer.pkl
- Prediction pipeline with neutral sentiment threshold logic
- Streamlit app for interactive inference

## Dataset

Expected dataset file:

- IMDB Dataset.csv

Expected columns:

- review
- sentiment

Sentiment values must be:

- positive
- negative

## Project Structure

- preprocess.py: Text cleaning and preprocessing utilities
- train.py: Training, evaluation, and model/vectorizer saving
- predict.py: Inference pipeline and CLI prediction
- app.py: Streamlit web app
- requirements.txt: Python dependencies
- model.pkl: Saved trained model (generated after training)
- vectorizer.pkl: Saved TF-IDF vectorizer (generated after training)

## Setup

### 1. Create and activate a Conda environment

```powershell
conda create -n nlp-sentiment python=3.11 -y
conda activate nlp-sentiment
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

## Train the Model

Run training and evaluation:

```powershell
python train.py
```

What this does:

- Loads and validates IMDB Dataset.csv
- Applies preprocessing to text
- Trains TF-IDF + Logistic Regression
- Prints evaluation metrics and confusion matrix
- Saves model.pkl and vectorizer.pkl

## Predict from Command Line

```powershell
python predict.py --text "This movie was amazing, emotional, and beautifully directed."
python predict.py --text "This was boring and a complete waste of time."
```

If --text is omitted, the script prompts for input.

## Neutral Sentiment Logic

Because the training dataset is binary, neutral is inferred from probability:

- Positive if probability > 0.6
- Negative if probability < 0.4
- Neutral otherwise

## Run Streamlit App

```powershell
streamlit run app.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

## Example Workflow

```powershell
conda activate nlp-sentiment
pip install -r requirements.txt
python train.py
python predict.py --text "Great movie with strong performances."
streamlit run app.py
```

## Troubleshooting

- Import errors in VS Code:
  - Select the correct Python interpreter (the nlp-sentiment environment)
- Missing model files in app/predict:
  - Run python train.py first
- NLTK resource errors:
  - The preprocessing module attempts automatic NLTK resource download on first run

## License

This project is for educational and learning purposes.

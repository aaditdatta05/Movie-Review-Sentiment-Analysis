"""Streamlit app for BERT-based sentiment analysis."""

from __future__ import annotations

import streamlit as st

from predict_bert import load_bert_model, predict_sentiment_bert


@st.cache_resource
def get_bert_model():
    """Load fine-tuned BERT model once per app session."""
    return load_bert_model()


def main() -> None:
    st.set_page_config(page_title="BERT Movie Review Sentiment", page_icon="🎬", layout="centered")
    st.title("Movie Review Sentiment Analysis (BERT)")
    st.write("Enter a movie review and click Predict to classify the sentiment using a fine-tuned BERT model.")

    review_text = st.text_area("Movie Review", height=180, placeholder="Type your review here...")
    predict_clicked = st.button("Predict")

    if not predict_clicked:
        return

    if not review_text.strip():
        st.warning("Please enter a review before predicting.")
        return

    try:
        model, tokenizer = get_bert_model()
        result = predict_sentiment_bert(review_text, model, tokenizer)
    except FileNotFoundError as error:
        st.error(
            "BERT model not found. Run python train_bert.py first to train and save the model."
        )
        st.caption(str(error))
        return
    except Exception as error:
        st.error("Prediction failed. Check logs for details.")
        st.caption(str(error))
        return

    st.subheader("Result")
    st.success(f"Sentiment: {result['label']}")
    st.write(f"Confidence Score: {result['confidence']:.2%}")
    st.write(f"Positive Class Probability: {result['positive_probability']:.2%}")


if __name__ == "__main__":
    main()

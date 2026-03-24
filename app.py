"""Streamlit app for movie review sentiment analysis."""

from __future__ import annotations

import streamlit as st

from predict import load_artifacts, predict_sentiment


@st.cache_resource
def get_artifacts():
    """Load trained artifacts once per app session."""
    return load_artifacts()


def main() -> None:
    st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬", layout="centered")
    st.title("Movie Review Sentiment Analysis")
    st.write("Enter a movie review and click Predict to classify the sentiment.")

    review_text = st.text_area("Movie Review", height=180, placeholder="Type your review here...")
    predict_clicked = st.button("Predict")

    if not predict_clicked:
        return

    if not review_text.strip():
        st.warning("Please enter a review before predicting.")
        return

    try:
        model, vectorizer = get_artifacts()
        result = predict_sentiment(review_text, model, vectorizer)
    except FileNotFoundError as error:
        st.error(
            "Model artifacts not found. Run train.py first to generate model.pkl and vectorizer.pkl."
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

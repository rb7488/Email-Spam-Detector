import streamlit as st
import re
import joblib


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)


# -----------------------------
# Load Saved Model
# -----------------------------
@st.cache_resource
def load_model():

    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

    return model, vectorizer


model, vectorizer = load_model()


# -----------------------------
# Text Preprocessing
# -----------------------------
def clean_text(text):

    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text


# -----------------------------
# Title
# -----------------------------
st.title("📧 Email Spam Detector")

st.write(
    "Machine Learning based Spam Detection using "
    "TF-IDF and Multinomial Naive Bayes."
)

st.divider()


# -----------------------------
# Email Input
# -----------------------------
st.subheader("Enter Email / Message")

message = st.text_area(
    "Type or paste your email/message below:",
    height=180,
    placeholder="Example: Congratulations! You've won a free iPhone. Click the link now!"
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Check for Spam", type="primary"):

    if message.strip() == "":
        st.warning("⚠️ Please enter a message first.")

    else:

        # Clean input
        cleaned_message = clean_text(message)

        # Convert text into TF-IDF features
        message_tfidf = vectorizer.transform(
            [cleaned_message]
        )

        # Prediction
        prediction = model.predict(
            message_tfidf
        )[0]

        # Probability
        probability = model.predict_proba(
            message_tfidf
        )[0]

        spam_probability = probability[1]

        st.divider()

        # Result
        if prediction == 1:

            st.error("🚨 SPAM EMAIL DETECTED")

            st.metric(
                "Spam Probability",
                f"{spam_probability:.2%}"
            )

        else:

            st.success("✅ NOT SPAM")

            st.metric(
                "Spam Probability",
                f"{spam_probability:.2%}"
            )


# -----------------------------
# Sample Messages
# -----------------------------
st.divider()

st.subheader("🧪 Try a Sample Message")

col1, col2 = st.columns(2)

with col1:

    if st.button("🚨 Spam Example"):

        st.code(
            "Congratulations! You've won a free iPhone. "
            "Click the link now!"
        )


with col2:

    if st.button("✅ Normal Example"):

        st.code(
            "Hey, are we meeting at 5 PM today?"
        )


# -----------------------------
# Model Information
# -----------------------------
st.divider()

st.caption(
    "Model: Multinomial Naive Bayes | "
    "Feature Extraction: TF-IDF"
)

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

        # Probability
        probability = model.predict_proba(message_tfidf)[0]
        spam_probability = probability[1]

        # Decision threshold
        threshold = 0.40

        if spam_probability >= threshold:
            prediction = 1
        else:
            prediction = 0

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

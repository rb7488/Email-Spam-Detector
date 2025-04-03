import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "message"]

# Convert labels to binary (spam = 1, not spam = 0)
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

df["message"] = df["message"].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label"], test_size=0.2, random_state=42)

# Convert text to numeric using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naïve Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Function to classify new messages
def classify_message(msg):
    msg_cleaned = clean_text(msg)
    msg_tfidf = vectorizer.transform([msg_cleaned])
    prediction = model.predict(msg_tfidf)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Test Example
test_message = "Congratulations! You've won a free iPhone. Click the link now!"
print(f"Message: {test_message}")
print(f"Classification: {classify_message(test_message)}")

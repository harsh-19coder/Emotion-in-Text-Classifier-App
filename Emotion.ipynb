import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load Dataset (Ensure your CSV file has 'Text' and 'Emotion' columns)
df = pd.read_csv("emotion_data.csv")

# Text Preprocessing Function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    
    text = text.lower()
    text = re.sub(r"\W+", " ", text)  # Remove special characters
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalnum()]
    
    return " ".join(tokens)

# Apply Preprocessing
df["Clean_Text"] = df["Text"].astype(str).apply(preprocess_text)

# Split Data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(df["Clean_Text"], df["Emotion"], test_size=0.2, random_state=42)

# Convert Text to TF-IDF Features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression Model
log_reg = LogisticRegression(max_iter=500, solver="liblinear")
log_reg.fit(X_train_tfidf, y_train)

# Evaluate the Model
y_pred = log_reg.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")  # Expecting ~80% accuracy
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save Model and Vectorizer for Future Use
with open("logistic_regression_model.pkl", "wb") as model_file:
    pickle.dump(log_reg, model_file)

with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("\nModel and vectorizer saved successfully!")

# ------------------------------ #
# 📌 User Input for Emotion Detection #
# ------------------------------ #

# Load Saved Model and Vectorizer
with open("logistic_regression_model.pkl", "rb") as model_file:
    log_reg = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Get User Input for Message
user_message = input("\nEnter your message: ")

# Process and Predict Emotion
processed_message = preprocess_text(user_message)
message_vector = vectorizer.transform([processed_message])
predicted_emotion = log_reg.predict(message_vector)[0]

# Display Result
print(f"\nPredicted Emotion: {predicted_emotion}")

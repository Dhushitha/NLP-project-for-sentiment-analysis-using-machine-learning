import pandas as pd
import nltk
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
df = pd.read_csv("IMDB Dataset.csv")

# Preprocessing function
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))   # ✅ LOAD ONCE

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]   # ✅ FAST
    return " ".join(words)

df['review'] = df['review'].apply(clean_text)

# Convert labels
df['sentiment'] = df['sentiment'].map({'positive':1, 'negative':0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Prediction
y_pred = model.predict(X_test_vec)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Test with custom input
def predict_sentiment(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    result = model.predict(vec)
    return "Positive 😊" if result[0] == 1 else "Negative 😡"

# Example
print(predict_sentiment("This movie was amazing and fantastic!"))
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", cm)

# Plot
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()




#for UI use these lines of code


import pickle

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
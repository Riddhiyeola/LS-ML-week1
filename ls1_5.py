import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

good_feedback = ["Excellent product", "Very satisfied", "Works perfectly", "Great value", "Highly recommend", "Loved it", "Top quality", "Amazing experience", "Very useful", "Absolutely fantastic"]

bad_feedback = ["Terrible quality", "Very disappointed", "Doesn't work", "Waste of money", "Bad experience", "Not recommended", "Cheap and useless", "Completely broke", "Very poor", "Worst ever"]

good = np.random.choice(good_feedback, 50)
bad = np.random.choice(bad_feedback, 50)

texts = np.concatenate([good, bad])
labels = ['good'] * 50 + ['bad'] * 50

df = pd.DataFrame({'text': texts, 'label': labels})
vectorizer = TfidfVectorizer(max_features=300, stop_words='english', lowercase=True)
X = vectorizer.fit_transform(df['text'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Precision:", precision_score(y_test, y_pred, pos_label='good'))
print("Recall:", recall_score(y_test, y_pred, pos_label='good'))
print("F1 Score:", f1_score(y_test, y_pred, pos_label='good'))

def text_preprocess_vectorize(texts, vectorizer):
    return vectorizer.transform(texts)
new_texts = ["This product is amazing", "Not worth the price"]
print("Predictions:", model.predict(text_preprocess_vectorize(new_texts, vectorizer)))

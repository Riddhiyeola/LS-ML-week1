import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import scipy
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
pph = ["An absolute masterpiece", "Loved the storyline", "Brilliant acting", "Amazing visuals", "Fantastic direction","Truly emotional", "A must-watch", "Well-paced and gripping", "Superb cinematography", "Touching and beautiful"]

nph = ["Terrible acting", "Poor storyline", "Completely boring", "Disappointing experience", "Too predictable","Bad direction", "Overhyped and dull", "Unrealistic and annoying", "Waste of time", "Worst movie ever"]
pr=np.random.choice(pph,size=50)
nr=np.random.choice(nph,size=50)
rev=np.concatenate([pr,nr])
sen=['positive']*50+['negative']*50
df = pd.DataFrame({
    'review': rev,
    'sentiment': sen
})
#print(df.head)

vect=CountVectorizer(max_features=500, stop_words='english')
a=vect.fit_transform(df['review'])
df1=pd.DataFrame(a.toarray(),columns=vect.get_feature_names_out())
#print(df1.head)

X_train, X_test, y_train, y_test = train_test_split(df1, df['sentiment'], test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", round(accuracy * 100, 2), "%")

def predict_review_sentiment(model, vectorizer, review):
    vec = vectorizer.transform([review])
    pred = model.predict(vec)
    return pred[0]
sample = "The movie was truly emotional and beautiful"
print("Review:", sample)
print("Predicted Sentiment:", predict_review_sentiment(model, vect, sample))
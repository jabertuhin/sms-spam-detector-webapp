import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.externals import joblib

# reading csv file
df = pd.read_csv("spam.csv", encoding="latin-1")

# dropping unnecesary columns.
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# changing column name
df.columns = ['label', 'message']

# changing label to 0 and 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Feature extraction with CountVectorizer
X = df['message']
y = df['label']
cv = CountVectorizer()
cv_fit = cv.fit(X)
X = cv.transform(X) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# saving the vector
pickle.dump(cv_fit, open("vector.pickel", "wb"))

#Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


# Saving the trained model
joblib.dump(clf, 'NB_spam_model.pkl')
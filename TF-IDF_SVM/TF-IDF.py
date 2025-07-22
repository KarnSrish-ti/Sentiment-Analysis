```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the vectorizer (you can tune max_features)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Fit on the training data and transform it
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Only transform the test data
X_test_tfidf = tfidf_vectorizer.transform(X_test)
```
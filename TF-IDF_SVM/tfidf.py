import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#Load the cleaned dataset (contains 'Target' and 'Cleaned')
df = pd.read_csv("../cleaned_dataset_lemmatized.csv")

#Extract input features and target labels
X = df['Cleaned']          
y = df['Target']           

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Define a TF-IDF vectorizer using character-level n-grams
tfidf = TfidfVectorizer(
    analyzer='char',         # Use character-level analysis
    ngram_range=(2, 4),      # Character 2-grams to 4-grams
    max_features=5000        # Limit to top 5000 features
)

# Fit the TF-IDF vectorizer on the training set and transform both sets
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Print out the shapes to verify success
print("Training vectorized shape:", X_train_tfidf.shape)
print("Testing vectorized shape:", X_test_tfidf.shape)

# Class Distribution
#why it matters? if classes are imbalanced, model might learn to predict only the majority class.
print("\nLabel Distribution:") 
print(df['Target'].value_counts()) 




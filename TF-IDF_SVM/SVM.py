import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# === Step 1: Load the dataset ===
df = pd.read_csv("../cleaned_dataset_lemmatized.csv")

# Optional: View column names and sample data
print("Available columns:", df.columns.tolist())
print(df.head())

# === Step 2: Select features and labels (FIXED) ===
X = df['Cleaned'].astype(str)   # Text input for TF-IDF, ensure it's string
y = df['Target']                # Target labels (e.g., 0/1)

# === Step 3: TF-IDF Vectorization ===
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# === Step 4: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# === Step 5: Train the Linear SVM Model ===
svm = LinearSVC(max_iter=10000)
svm.fit(X_train, y_train)

# === Step 6: Evaluate the Model ===
y_pred = svm.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

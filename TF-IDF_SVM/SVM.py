from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# X_tfidf → your TF-IDF features
# y → your target labels

# Step 1: Split TF-IDF features and labels
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

# Step 2: Train SVM model
svm = LinearSVC()
svm.fit(X_train, y_train)

# Step 3: Predict and Evaluate
y_pred = svm.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

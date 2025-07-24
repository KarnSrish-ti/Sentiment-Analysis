# Import all required libraries
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# --- DYNAMICALLY CONSTRUCT THE FILE PATH ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
csv_path = os.path.join(parent_dir, 'cleaned_dataset_lemmatized.csv')

# --- LOAD THE DATASET ---
print(f"Attempting to load dataset from: {csv_path}")
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"\n--- ERROR: FILE NOT FOUND at {csv_path} ---")
    # ... (error message from before) ...
    exit()

df['Cleaned'] = df['Cleaned'].fillna('')
X = df['Cleaned']
y = df['Target']

# --- SPLIT DATA ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- FEATURE EXTRACTION (TF-IDF) ---
print("\nCreating character-level TF-IDF features...")
tfidf = TfidfVectorizer(
    analyzer='char', ngram_range=(2, 4), max_features=5000
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print("Training vectorized shape:", X_train_tfidf.shape)
print("Testing vectorized shape:", X_test_tfidf.shape)

# --- MODEL TRAINING (SVM) ---
print("\nTraining Support Vector Machine (SVM) model...")
model = SVC(kernel='linear', class_weight='balanced', random_state=42)
model.fit(X_train_tfidf, y_train)

# --- EVALUATION ---
print("\nEvaluating model on the test set:")
y_pred = model.predict(X_test_tfidf)

# --- SAVE CLASSIFICATION REPORT ---
# This is your key result with precision, recall, f1-score
report = classification_report(y_test, y_pred)
print(report) # Print report to the console

# Define the path to save the report (it will be in the TF-IDF_SVM folder)
report_path = os.path.join(script_dir, 'classification_report_tfidf_svm.txt')
with open(report_path, 'w') as f:
    f.write("Classification Report for TF-IDF + SVM Model\n")
    f.write("="*50 + "\n")
    f.write(report)
print(f"\nClassification report saved to: {report_path}")


# --- GENERATE AND SAVE CONFUSION MATRIX ---
# This shows a breakdown of correct vs. incorrect predictions
conf_matrix = confusion_matrix(y_test, y_pred)

# Create a heatmap for visualization
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix for TF-IDF + SVM')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

# Define the path to save the image (it will also be in the TF-IDF_SVM folder)
matrix_path = os.path.join(script_dir, 'confusion_matrix_tfidf_svm.png')
plt.savefig(matrix_path)
print(f"Confusion matrix image saved to: {matrix_path}")


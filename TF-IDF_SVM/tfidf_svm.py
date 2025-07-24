# Import all required libraries
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt #for data visualization
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# --- DYNAMICALLY CONSTRUCT THE FILE PATH ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
csv_path = os.path.join(parent_dir, 'cleaned_dataset_lemmatized.csv')

# LOAD THE DATASET 
print(f"Attempting to load dataset from: {csv_path}")
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"\nERROR: FILE NOT FOUND at {csv_path} ")
    #  (error message from before) 
    exit()

df['Cleaned'] = df['Cleaned'].fillna('')
X = df['Cleaned']
y = df['Target']

# SPLIT DATA 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#  FEATURE EXTRACTION (TF-IDF)
print("\nCreating character-level TF-IDF features...")
tfidf = TfidfVectorizer(
    analyzer='char', ngram_range=(2, 4), max_features=5000
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print("Training vectorized shape:", X_train_tfidf.shape)
print("Testing vectorized shape:", X_test_tfidf.shape)

#  MODEL TRAINING (SVM)
print("\nTraining Support Vector Machine (SVM) model...")
model = SVC(kernel='linear', class_weight='balanced', random_state=42)
model.fit(X_train_tfidf, y_train)

# EVALUATION 
print("\nEvaluating model on the test set:")
y_pred = model.predict(X_test_tfidf)

# SAVE CLASSIFICATION REPORT 
#result with precision, recall, f1-score
report = classification_report(y_test, y_pred)
print(report) # Print report to the console

# Define the path to save the report (it will be in the TF-IDF_SVM folder)
report_path = os.path.join(script_dir, 'classification_report_tfidf_svm.txt')
with open(report_path, 'w') as f:
    f.write("Classification Report for TF-IDF + SVM Model\n")
    f.write("="*50 + "\n")
    f.write(report)
print(f"\nClassification report saved to: {report_path}")


# GENERATE AND SAVE CONFUSION MATRIX 
# --- VISUALIZATION (Confusion Matrix) ---
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Generate the confusion matrix from the test labels and predictions
cm = confusion_matrix(y_test, y_pred)

# Define the class names in the correct order (0=Positive, 1=Negative)
class_names = ['Positive', 'Negative']

# Create a figure and axes
fig, ax = plt.subplots(figsize=(8, 6))

# Create the heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)

# Set the labels for the axes
ax.set_xlabel('Predicted Label')
ax.set_ylabel('Actual Label')
ax.set_title('Confusion Matrix for TF-IDF + SVM')

# Set the tick labels to our class names
ax.xaxis.set_ticklabels(['Predicted ' + name for name in class_names])
ax.yaxis.set_ticklabels(['Actual ' + name for name in class_names])

# Show the plot
plt.show()

# To save the figure to a file for your report:
# fig.savefig("confusion_matrix_tfidf_svm.png")

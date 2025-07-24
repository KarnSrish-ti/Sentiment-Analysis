# Import all required libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Import TensorFlow and Keras components
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D

# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# --- DATA LOADING ---
df = pd.read_csv("cleaned_dataset_lemmatized.csv")
df['Cleaned'] = df['Cleaned'].fillna('')

# Extract input features and target labels
X = df['Cleaned']          
y = df['Target'] 

# Split data before any processing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- LSTM DATA PREPARATION ---
# 1. Tokenize the text (convert words to unique integer IDs)
vocab_size = 10000 # Keep the top 10,000 most frequent words
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>") # <OOV> handles words not in vocabulary
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# 2. Pad sequences to ensure every sequence has the same length
max_length = 200 # Max number of words in a sequence
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

# --- BUILD THE LSTM MODEL ---
print("\n--- Building the LSTM Model ---")
embedding_dim = 128
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    SpatialDropout1D(0.3), # Dropout layer to prevent overfitting
    LSTM(64, dropout=0.3, recurrent_dropout=0.3),
    Dense(1, activation='sigmoid') # Sigmoid activation for binary (0 or 1) output
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# --- TRAIN THE LSTM MODEL ---
# Note: This step might take a few minutes
print("\n--- Training the LSTM Model ---")
num_epochs = 5
batch_size = 32

history = model.fit(
    X_train_pad, y_train,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(X_test_pad, y_test),
    verbose=2 # Show progress for each epoch
)

# --- EVALUATE AND SAVE RESULTS ---
print("\n--- Evaluating LSTM Model ---")
# Make predictions (these will be probabilities)
y_pred_prob = model.predict(X_test_pad)
# Convert probabilities to class labels (0 or 1) based on a 0.5 threshold
y_pred = (y_pred_prob > 0.5).astype("int32")

# 1. GENERATE AND SAVE THE CLASSIFICATION REPORT
report_str = classification_report(y_test, y_pred, target_names=['Positive (0)', 'Negative (1)'])
print(report_str)

report_filename = 'lstm_classification_report.txt'
with open(report_filename, 'w', encoding='utf-8') as f:
    f.write("Classification Report for LSTM Model\n")
    f.write("="*50 + "\n")
    f.write(report_str)
print(f"Classification report saved to: {report_filename}")

# 2. GENERATE AND SAVE THE CONFUSION MATRIX PLOT
cm = confusion_matrix(y_test, y_pred)
class_names = ['Positive', 'Negative']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Predicted ' + name for name in class_names],
            yticklabels=['Actual ' + name for name in class_names])

plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix for LSTM Model')

matrix_filename = 'lstm_confusion_matrix.png'
plt.savefig(matrix_filename)
print(f"Confusion matrix image saved to: {matrix_filename}")
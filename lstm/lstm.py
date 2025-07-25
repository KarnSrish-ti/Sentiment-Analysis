import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam # Import Adam optimizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Main Execution ---
if __name__ == "__main__":
    # --- 1. Configuration and Data Loading ---
    FILE_PATH = '../cleaned_dataset_lemmatized.csv'
    VOCAB_SIZE = 10000  # Max number of words to keep
    MAX_LENGTH = 120    # Max length of sequences
    EMBEDDING_DIM = 100  # Embedding vector dimension
    TEST_SPLIT = 0.2

    # Load the dataset
    try:
        df = pd.read_csv(FILE_PATH)
        df.columns = ['target', 'cleaned']
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        exit()

    df.dropna(subset=['cleaned', 'target'], inplace=True)

    # --- 2. Data Preprocessing ---
    sentences = df['cleaned'].astype(str).values
    labels = df['target'].values

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

    # --- 3. Split Data into Training and Testing Sets ---
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, labels, test_size=TEST_SPLIT, random_state=42
    )
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # --- 4. Build and Compile the Tuned LSTM Model ---
    model = Sequential()
    model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LENGTH))
    model.add(SpatialDropout1D(0.25))
    model.add(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(Bidirectional(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))

    # CRITICAL CHANGE: Explicitly define the optimizer and set the learning rate.
    optimizer = Adam(learning_rate=0.0005)

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    model.summary()

    # --- 5. Train the Model ---
    print("\nStarting model training...")
    # INCREASED EPOCHS: Give the model more time to learn.
    EPOCHS = 30
    BATCH_SIZE = 128

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=2
    )
    print("Model training completed.\n")

    # --- 6. Evaluate and Predict ---
    results = model.evaluate(X_test, y_test, verbose=0)
    print("--- LSTM Model Evaluation ---")
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]*100:.2f}%")
    print(f"Test Precision: {results[2]:.4f}")
    print(f"Test Recall: {results[3]:.4f}\n")

    # --- 7. Classification Report and Confusion Matrix ---
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=['Positive (0)', 'Negative (1)'])
    print(report)
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive (0)', 'Negative (1)'], yticklabels=['Positive (0)', 'Negative (1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    # --- 8. Plot Training History ---
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.close()

    def predict_sentiment(text):
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post', truncating='post')
        prediction = model.predict(padded_sequence, verbose=0)[0][0]
        sentiment = "Negative (1)" if prediction > 0.5 else "Positive (0)"
        print(f"Text: '{text}'")
        print(f"Prediction Score: {prediction:.4f} -> {sentiment}")

    # Call the function with your test sentences
    predict_sentiment("उपस्थित विद्वान कस कस गुठी")
    predict_sentiment("गुठी विधेक ल्याएर ठमेल मा राज गुठि को जग्गा मा बने को छाया सेन्टर जस्ता लाई जोगाउन को लागि ल्याउदैछ विधेक ।")

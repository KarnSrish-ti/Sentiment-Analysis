import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from sklearn.model_selection import train_test_split

# Main Execution
if __name__ == "__main__":
    # --- 1. Configuration and Data Loading ---
    FILE_PATH = '../cleaned_dataset_lemmatized.csv'
    VOCAB_SIZE = 15000     # Increase vocabulary size
    MAX_LENGTH = 120       # Increase sequence length slightly if sentences are long
    EMBEDDING_DIM = 128    # Increase the embedding dimension
    TEST_SPLIT = 0.2    # 20% of data will be used for testing

    # Load the dataset
    try:
        # Assuming the CSV might not have a header, we name the columns
        df = pd.read_csv(FILE_PATH)
        # Ensure the column names are correct as per your description
        df.columns = ['target', 'cleaned']
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file '{FILE_PATH}' was not found.")
        print("Please make sure the file exists and the path is correct.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        exit()

    # Drop any rows with missing values to be safe
    df.dropna(subset=['cleaned', 'target'], inplace=True)

    # --- 2. Data Preprocessing ---
    # Separate features (text) and labels (target)
    sentences = df['cleaned'].astype(str).values
    labels = df['target'].values

    # Tokenize the text: convert words to integers
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(sentences)

    # Pad the sequences so they all have the same length
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

    # --- 3. Split Data into Training and Testing Sets ---
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences,
        labels,
        test_size=TEST_SPLIT,
        random_state=42 # for reproducibility
    )

    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    model = Sequential()
    model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LENGTH))
    model.add(SpatialDropout1D(0.3)) # Slightly increase dropout
# Increase LSTM units and make it return sequences for stacking
    model.add(LSTM(units=128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
# Add a second LSTM layer
    model.add(LSTM(units=64, dropout=0.3, recurrent_dropout=0.3))
# Add a dense layer before the output
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile the model for a binary classification task
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary() # Print a summary of the model architecture

    # --- 5. Train the Model ---
    print("\nStarting model training...")
    EPOCHS = 10
    BATCH_SIZE = 64
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=2
    )
    print("Model training completed.\n")

    # --- 6. Evaluate the Model ---
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("--- Model Evaluation ---")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Test Loss: {loss:.4f}\n")

    # --- 7. Prediction Example ---
    # 0 = positive, 1 = negative
    sample_text_positive = "उपस्थित विद्वान कस कस गुठी"
    sample_text_negative = "गुठी विधेक ल्याएर ठमेल मा राज गुठि को जग्गा मा बने को छाया सेन्टर जस्ता लाई जोगाउन को लागि ल्याउदैछ विधेक ।"

    def predict_sentiment(text):
        # Preprocess the new text just like the training data
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post', truncating='post')
        # Make a prediction
        prediction = model.predict(padded_sequence)[0][0]
        sentiment = "Negative" if prediction > 0.5 else "Positive"
        print(f"Text: '{text}'")
        print(f"Prediction Score: {prediction:.4f}")
        print(f"Predicted Sentiment: {sentiment} (0=Positive, 1=Negative)\n")

    predict_sentiment(sample_text_positive)
    predict_sentiment(sample_text_negative)
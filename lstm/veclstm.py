# lstm_model.py
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from vectorizer import load_dataset, train_word2vec, compute_tfidf, document_vector

def create_embedding_matrix(w2v_model, idf_scores, word_index, embedding_dim):
    """
    Create embedding matrix with TF-IDF weighted Word2Vec vectors
    
    Args:
        w2v_model: Trained Word2Vec model
        idf_scores: Dictionary of word IDF scores
        word_index: Tokenizer word index
        embedding_dim: Dimension of embedding vectors
        
    Returns:
        numpy.ndarray: Embedding matrix
    """
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word, i in word_index.items():
        if word in w2v_model.wv and word in idf_scores:
            # Multiply Word2Vec vector by TF-IDF weight
            embedding_matrix[i] = w2v_model.wv[word] * idf_scores[word]
    return embedding_matrix

def build_lstm_model(embedding_matrix, max_length, num_classes):
    """
    Build LSTM model architecture
    
    Args:
        embedding_matrix: Pre-trained embedding matrix
        max_length: Maximum sequence length
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # Add embedding layer with pre-trained weights
    model.add(Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=max_length,
        trainable=False  # Freeze embeddings
    ))
    
    # Add LSTM layers
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    try:
        # Load and preprocess data
        dataset_path = "../cleaned_dataset_lemmatized.csv"
        texts, labels = load_dataset(dataset_path)
        
        # Train Word2Vec and compute TF-IDF
        tokenized_texts = [text.split() for text in texts]
        w2v_model = train_word2vec(tokenized_texts)
        idf_scores = compute_tfidf(texts)
        
        # Create numerical sequences
        from tensorflow.keras.preprocessing.text import Tokenizer
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        max_length = max(len(seq) for seq in sequences)
        X = pad_sequences(sequences, maxlen=max_length, padding='post')
        
        # Create embedding matrix
        embedding_dim = w2v_model.vector_size
        embedding_matrix = create_embedding_matrix(
            w2v_model, idf_scores, tokenizer.word_index, embedding_dim)
        
        # Prepare labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(labels)
        num_classes = len(le.classes_)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Build and train model
        model = build_lstm_model(embedding_matrix, max_length, num_classes)
        
        early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=20,
            batch_size=64,
            callbacks=[early_stopping]
        )
        
        # Evaluate
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
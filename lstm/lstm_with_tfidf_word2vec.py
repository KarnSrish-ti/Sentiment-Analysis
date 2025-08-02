#import necessary libraries
#for operating system interction
import os
#for data manipulation and analysis
import pandas as pd
#for numerical operations and array handeling
import numpy as np
#for training and embedding models
from gensim.models import Word2Vec
#for TF-IDF calculations
from sklearn.feature_extraction.text import TfidfVectorizer
#for splitting the datasets
from sklearn.model_selection import train_test_split
#support vector machine classifier
from sklearn.svm import SVC
#evaluation metrics
from sklearn.metrics import classification_report, accuracy_score
#exception for untrained models
from sklearn.exceptions import NotFittedError
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop

def load_dataset(path):
    """
    Load and validate the dataset from a CSV file.
    
    Args:
        path (str): Path to the CSV file containing the dataset
        
    Returns:
        tuple: (list of texts, list of corresponding labels)
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If required columns are missing
    """
    # Check if the file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f" File not found: {path}")
    # Read CSV file into dataframe
    df = pd.read_csv(path)
    # Validate required columns exist
    if "Cleaned" not in df.columns or "Target" not in df.columns:
        raise ValueError(" Columns 'Cleaned' and 'Target' must exist in the dataset.")
    # Remove rows with missing values in key columns
    df = df.dropna(subset=["Cleaned", "Target"])
    # Return texts and labels as lists of strings
    return df["Cleaned"].astype(str).tolist(), df["Target"].astype(str).tolist()

def train_word2vec(tokenized_texts, vector_size=100):
    """
    Train a Word2Vec model on tokenized text data. 
    
    Args:
        tokenized_texts (list): List of tokenized sentences (list of words)
        vector_size (int): Dimensionality of word vectors
        
    Returns:
        Word2Vec: Trained Word2Vec model
    """
    print("🔧 Training Word2Vec model...")
    # Initialize and train Word2Vec model
    # - sentences: The tokenized text data
    # - vector_size: Size of word vectors (default 100)
    # - window: Maximum distance between current and predicted word
    # - min_count: Ignores words with frequency lower than this
    # - workers: Number of CPU cores to use
    return Word2Vec(sentences=tokenized_texts, vector_size=vector_size, window=5, min_count=1, workers=4)

def compute_tfidf(texts):
    """
    Compute TF-IDF scores for all words in the corpus.
    
    Args:
        texts (list): List of raw text documents
        
    Returns:
        dict: Dictionary mapping words to their IDF scores
    """
    print("Computing TF-IDF scores...")
    # Initialize tf-idf vectorizer
    vectorizer = TfidfVectorizer()
    # Learn vocabulary and TF-IDF vectorizer
    vectorizer.fit(texts)
    # Create dictionary of word: idf_scores pairs
    return dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

def document_vector(doc, w2v_model, idf_scores):
    """
    Convert a document to a weighted vector using Word2Vec and TF-IDF.
    
    Args:
        doc (str): Input document text
        w2v_model (Word2Vec): Trained Word2Vec model
        idf_scores (dict): Precomputed IDF scores for words
        
    Returns:
        numpy.ndarray: Weighted document vector
    """
    # Split document into words
    words = doc.split()
    # Initialize zero vector of same size as word vectors
    vec = np.zeros(w2v_model.vector_size)
    # Initialize sum of weights for normalization
    weight_sum = 0.0

    # For each word in the document
    for word in words:
        # Check if word exists in both word2vec model and IDF scores
        if word in w2v_model.wv and word in idf_scores:
            # Get idf weight for word
            weight = idf_scores[word]
            # Add weighted word vector 
            vec += w2v_model.wv[word] * weight
            # Accumulate weights for normalization
            weight_sum += weight

    # Return normalized vector
    return vec / weight_sum if weight_sum > 0 else vec

def vectorize_documents(texts, w2v_model, idf_scores):
    """
    Convert all documents to weighted vector representations.
    
    Args:
        texts (list): List of document strings
        w2v_model (Word2Vec): Trained Word2Vec model
        idf_scores (dict): Precomputed IDF scores
        
    Returns:
        numpy.ndarray: Matrix of document vectors
    """
    print(" Vectorizing documents...")
    # Convert each document to vector and stack into matrix
    return np.array([document_vector(text, w2v_model, idf_scores) for text in texts])


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam

def create_embedding_matrix(w2v_model, idf_scores, word_index, embedding_dim):
    # Create embedding matrix with TF-IDF weighted Word2Vec vectors
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        if word in w2v_model.wv and word in idf_scores:
            embedding_matrix[i] = w2v_model.wv[word] * idf_scores[word]
    return embedding_matrix

def train_and_evaluate_lstm(texts, labels, w2v_model, idf_scores):
    # Prepare data for LSTM
    print("Preparing data for LSTM...")
    # Tokenize the texts
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    # Convert texts to sequences of word indices
    sequences = tokenizer.texts_to_sequences(texts)
    # Set a fixed maximum sequence length for padding
    max_length = 150
    # Pad sequences to the same length
    X = pad_sequences(sequences, maxlen=max_length, padding='post')

    # Encode labels as integers
    le = LabelEncoder()
    y = le.fit_transform(labels)
    num_classes = len(le.classes_)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create the embedding matrix using Word2Vec and TF-IDF
    embedding_dim = w2v_model.vector_size
    embedding_matrix = create_embedding_matrix(w2v_model, idf_scores, tokenizer.word_index, embedding_dim)

    # Build the LSTM model
    model = Sequential()
    # Embedding layer with pre-trained weights
    model.add(Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=max_length,
        trainable=True  # Allow embedding weights to be updated
    ))
    # First Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Dropout(0.3))
    # Second Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.3))
    # Output layer
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))
    else:
        model.add(Dense(num_classes, activation='softmax'))

    # Compile the model with optimizer, loss, and metrics
    model.compile(
        optimizer=RMSprop(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(patience=6, restore_best_weights=True)
    print("Training LSTM model...")
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=2
    )

    # Evaluate the model on the test set
    print("\nEvaluating LSTM model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    # Predict classes for the test set
    y_pred = np.argmax(model.predict(X_test), axis=1) if num_classes > 2 else (model.predict(X_test) > 0.5).astype(int).flatten()
    # Print classification report
    print("Classification Report:\n", classification_report(y_test, y_pred))


def main():
    """
    Main execution function that orchestrates the entire pipeline using LSTM.
    """
    try:
        # Path to the cleaned dataset
        dataset_path = "../cleaned_dataset_lemmatized.csv"
        # Load and validate dataset
        texts, labels = load_dataset(dataset_path)
        # Tokenize texts for Word2Vec training
        tokenized_texts = [text.split() for text in texts]
        # Train Word2Vec model
        w2v_model = train_word2vec(tokenized_texts)
        # Compute TF-IDF scores
        idf_scores = compute_tfidf(texts)
        # Train and evaluate the LSTM model
        train_and_evaluate_lstm(texts, labels, w2v_model, idf_scores)
    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except ValueError as val_error:
        print(val_error)
    except NotFittedError as fit_error:
        print(" Model error:", fit_error)
    except Exception as e:
        print(" An unexpected error occurred:", e)

if __name__ == "__main__":
    main()

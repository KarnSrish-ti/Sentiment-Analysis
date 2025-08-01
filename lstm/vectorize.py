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
    #checks if the file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f" File not found: {path}")
    #read CSV file into dataframe
    df = pd.read_csv(path)
    #validate required columns exist
    if "Cleaned" not in df.columns or "Target" not in df.columns:
        raise ValueError(" Columns 'Cleaned' and 'Target' must exist in the dataset.")
    #remove rows with missing values n key columns
    df = df.dropna(subset=["Cleaned", "Target"])
    #return texts and labels as lists of strings
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
    # Initialize and train Word2Vec model with these parameters:
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
    #initialize tf-idf vectorizer
    vectorizer = TfidfVectorizer()
    #learn vocabulary and TF-IDF vectorizer
    vectorizer.fit(texts)
    #create dictionalry of word: idf_scores pairs
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
    #split document into words
    words = doc.split()
    #initialize zero vector of same size as word vectors
    vec = np.zeros(w2v_model.vector_size)
    #initialize sum of weights for normalization
    weight_sum = 0.0

    #for each word in the document
    for word in words:
        #check if word exists in both word2vec model and IDF scores
        if word in w2v_model.wv and word in idf_scores:
            #get idf weight for word
            weight = idf_scores[word]
            #add weighted word vector 
            vec += w2v_model.wv[word] * weight
            #accumulate weights for normalization
            weight_sum += weight

    #return normalized vector
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
    #convert each document to vector and stack into matrix
    return np.array([document_vector(text, w2v_model, idf_scores) for text in texts])

def train_and_evaluate_model(X, y):
    """
    Train SVM classifier and evaluate performance.
    
    Args:
        X (numpy.ndarray): Feature matrix (document vectors)
        y (numpy.ndarray): Target labels
    """ 
    print(" Splitting dataset and training SVM...")
    #split data into 80% train and 20% test with startified samplng
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    #initialize svm classifier with linear kernel
    svm = SVC(kernel='linear', C=1.0)
    #train model for training data
    svm.fit(X_train, y_train)

    #make prediction on test set
    y_pred = svm.predict(X_test)
    #print evaluation metrices
    print("\n Evaluation Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

def main():
    """
    Main execution function that orchestrates the entire pipeline.
    """
    try:
        #path to dataset file
        dataset_path = "../cleaned_dataset_lemmatized.csv"
        #load and validate dataset
        texts, labels = load_dataset(dataset_path)
        #tokenize texts
        tokenized_texts = [text.split() for text in texts]

        #train word2vec model on tokenized texts
        w2v_model = train_word2vec(tokenized_texts)
        #compute ifd scores for all words
        idf_scores = compute_tfidf(texts)
        #covert documents into wieghted vectors
        X = vectorize_documents(texts, w2v_model, idf_scores)
        #convert labels to numpy array
        y = np.array(labels)

        #train and evaluate SVM model
        train_and_evaluate_model(X, y)

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

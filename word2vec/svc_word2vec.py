# Import required libraries
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import datetime
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_output_dir():
    """Create output directory with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_and_validate_data(filepath):
    """Load and validate the dataset."""
    try:
        df = pd.read_csv(filepath)
        logger.info("Dataset loaded successfully")
        
        # Validate required columns
        required_columns = ["Cleaned", "Target"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain columns: {required_columns}")
            
        logger.info(f"Available columns: {df.columns.tolist()}")
        logger.info("\nClass distribution:\n%s", df["Target"].value_counts())
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def preprocess_tokens(df):
    """Convert text data to tokens."""
    try:
        # First attempt: direct split if already tokenized
        if isinstance(df["Cleaned"].iloc[0], list):
            df["Tokens"] = df["Cleaned"]
        elif isinstance(df["Cleaned"].iloc[0], str):
            try:
                # Try evaluating as list literal
                df["Tokens"] = df["Cleaned"].apply(ast.literal_eval)
            except:
                # Fallback to string splitting
                df["Tokens"] = df["Cleaned"].str.split()
        return df
    except Exception as e:
        logger.error(f"Error in token conversion: {str(e)}")
        raise

def train_word2vec(tokens, output_dir, vector_size=300, window=7, min_count=5, workers=4, sg=1):
    """Train Word2Vec model and save it."""
    try:
        model = Word2Vec(
            sentences=tokens,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg
        )
        logger.info("Word2Vec model trained successfully!")
        logger.info("Vocabulary size: %d words", len(model.wv))
        
        # Save Word2Vec model
        w2v_path = os.path.join(output_dir, "word2vec.model")
        model.save(w2v_path)
        logger.info("Word2Vec model saved to: %s", w2v_path)
        
        return model
    except Exception as e:
        logger.error(f"Error training Word2Vec: {str(e)}")
        raise

def document_vector(tokens, model):
    """Convert tokens to document vector by averaging word vectors."""
    try:
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
    except Exception as e:
        logger.error(f"Error creating document vector: {str(e)}")
        raise

def train_and_evaluate(X, y, output_dir):
    """Train SVM classifier, evaluate performance, and save everything."""
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        clf = SVC(kernel='rbf', class_weight='balanced')
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred)
        
        # Save classification report
        report_path = os.path.join(output_dir, "classification_report.txt")
        with open(report_path, 'w') as f:
            f.write(f"Classification Report\n")
            f.write("====================\n\n")
            f.write(report)
            f.write("\n\nModel Parameters:\n")
            f.write(str(clf.get_params()))
        logger.info("Classification report saved to: %s", report_path)

        
        
        # Save SVM model
        svm_path = os.path.join(output_dir, "svm_classifier.joblib")
        joblib.dump(clf, svm_path)
        logger.info("SVM classifier saved to: %s", svm_path)
        
        return clf
    except Exception as e:
        logger.error(f"Error in modeling: {str(e)}")
        raise

def main():
    try:
        # Create output directory
        output_dir = create_output_dir()
        logger.info("Output will be saved to: %s", output_dir)
        
        # 1. Data Loading
        df = load_and_validate_data("../cleaned_dataset_lemmatized.csv")
        
        # 2. Token Preprocessing
        df = preprocess_tokens(df)
        
        # 3. Word Embedding
        w2v_model = train_word2vec(df["Tokens"], output_dir)
        
        # 4. Document Vectorization
        X = np.array([document_vector(tokens, w2v_model) for tokens in df["Tokens"]])
        y = df["Target"]
        
        # 5. Model Training and Evaluation
        classifier = train_and_evaluate(X, y, output_dir)
        
        logger.info("Process completed successfully! All outputs saved to: %s", output_dir)
        return classifier, w2v_model
        
    except Exception as e:
        logger.error("Process failed: %s", str(e))
        return None, None

if __name__ == "__main__":
    classifier, w2v_model = main()

    
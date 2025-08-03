import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib
import time

def load_data(filepath, text_col='Cleaned', target_col='Target'):
    """Load and validate dataset"""
    df = pd.read_csv(filepath)
    assert {text_col, target_col}.issubset(df.columns), "Missing required columns"
    return df[text_col], df[target_col]

def train_model(X, y, test_size=0.2, random_state=42, save_model=False):
    """
    Train and evaluate Naive Bayes classifier with optimized pipeline
    Args:
        X: Text data
        y: Labels
        test_size: Size of test set
        random_state: Random seed
        save_model: Whether to save the trained model
    Returns:
        Trained model and evaluation metrics
    """
    # Split data with stratification to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Create optimized pipeline with TF-IDF vectorizer
    pipeline = make_pipeline(
        TfidfVectorizer(
            max_features=10000,        # Limit vocabulary size
            ngram_range=(1, 2),       # Use uni and bi-grams
            stop_words='english',      # Remove stop words
            min_df=5,                 # Ignore terms with low doc frequency
            max_df=0.7                # Ignore terms with high doc frequency
        ),
        MultinomialNB(alpha=0.1)      # Additive smoothing parameter
    )
    
    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'tfidfvectorizer__max_features': [5000, 10000, 20000],
        'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
        'multinomialnb__alpha': [0.01, 0.1, 1.0]
    }
    
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=3, 
        n_jobs=-1, 
        verbose=1,
        scoring='accuracy'
    )
    
    print("Training model with hyperparameter tuning...")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\nBest Parameters:", grid_search.best_params_)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)
    
    # Save model if requested
    if save_model:
        model_filename = 'optimized_nb_model.joblib'
        joblib.dump(best_model, model_filename)
        print(f"\nModel saved as {model_filename}")
    
    return best_model, {'accuracy': accuracy, 'report': report, 'confusion_matrix': cm}

if __name__ == "__main__":
    # Load data
    X, y = load_data('../cleaned_dataset_lemmatized.csv')
    
    # Train and evaluate model
    model, metrics = train_model(X, y, save_model=True)
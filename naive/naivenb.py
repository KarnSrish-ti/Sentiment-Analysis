import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.exceptions import NotFittedError
import joblib
import time
import logging
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('text_classifier.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TextClassifier:
    def __init__(self, model_path='optimized_nb_model.joblib'):
        self.model = None
        self.vectorizer = None
        self.model_path = Path(model_path)
        self.is_trained = False

    def load_data(self, filepath, text_col='Cleaned', target_col='Target'):
        """Load and validate dataset with comprehensive error handling"""
        try:
            if not Path(filepath).exists():
                raise FileNotFoundError(f"Data file not found at {filepath}")
            
            logger.info(f"Loading data from {filepath}")
            df = pd.read_csv(filepath)
            
            # Validate columns
            missing_cols = {text_col, target_col} - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Check for empty data
            if df.empty:
                raise ValueError("Loaded an empty DataFrame")
                
            # Validate target values
            if not set(df[target_col].unique()).issubset({0, 1}):
                raise ValueError("Target column should contain only 0 and 1")
                
            return df[text_col], df[target_col]
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _create_pipeline(self):
        """Create the model pipeline with default parameters"""
        return make_pipeline(
            TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=5,
                max_df=0.7
            ),
            MultinomialNB(alpha=0.1)
        )

    def train(self, X, y, test_size=0.2, random_state=42, save_model=True):
        """Train the model with comprehensive error handling and logging"""
        try:
            # Input validation
            if len(X) != len(y):
                raise ValueError("X and y must have the same length")
            if test_size <= 0 or test_size >= 1:
                raise ValueError("test_size must be between 0 and 1")
                
            logger.info("Starting model training...")
            start_time = time.time()
            
            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state, 
                stratify=y
            )
            
            # Create pipeline
            pipeline = self._create_pipeline()
            
            # Hyperparameter grid
            param_grid = {
                'tfidfvectorizer__max_features': [5000, 10000, 20000],
                'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
                'multinomialnb__alpha': [0.01, 0.1, 1.0]
            }
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=3,
                n_jobs=-1,
                verbose=1,
                scoring='accuracy',
                error_score='raise'
            )
            
            logger.info("Performing grid search...")
            grid_search.fit(X_train, y_train)
            
            # Store the best model
            self.model = grid_search.best_estimator_
            self.is_trained = True
            
            # Evaluate on test set
            evaluation = self.evaluate(X_test, y_test)
            
            logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Test accuracy: {evaluation['accuracy']:.4f}")
            
            if save_model:
                self.save_model()
                
            return evaluation
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def evaluate(self, X, y):
        """Evaluate model performance with enhanced visualization"""
        if not self.is_trained:
            raise NotFittedError("Model must be trained before evaluation")
            
        try:
            y_pred = self.model.predict(X)
            
            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            report = classification_report(y, y_pred)
            cm = confusion_matrix(y, y_pred)
            
            # Create accuracy table
            accuracy_table = [
                ["Metric", "Value"],
                ["Accuracy", f"{accuracy:.4f}"],
                ["Precision", f"{precision_score(y, y_pred):.4f}"],
                ["Recall", f"{recall_score(y, y_pred):.4f}"],
                ["F1-Score", f"{f1_score(y, y_pred):.4f}"]
            ]
            
            print("\n" + "="*50)
            print("MODEL PERFORMANCE METRICS".center(50))
            print("="*50)
            print(tabulate(accuracy_table, headers="firstrow", tablefmt="grid"))
            print("\nCLASSIFICATION REPORT:")
            print(report)
            
            # Plot confusion matrix
            self._plot_confusion_matrix(cm)
            
            return {
                'accuracy': accuracy,
                'report': report,
                'confusion_matrix': cm,
                'y_true': y,
                'y_pred': y_pred
            }
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise

    def _plot_confusion_matrix(self, cm):
        """Plot a beautiful confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        
        # Save and show the plot
        plt.savefig('confusion_matrix.png')
        plt.show()
        
        # Print ASCII version in console
        print("\nCONFUSION MATRIX (ASCII):")
        print(f"""
        Predicted
        ---------------
        | {cm[0][0]} | {cm[0][1]} |  Negative
        ---------------
        | {cm[1][0]} | {cm[1][1]} |  Positive
        ---------------
        Actual
        """)

    def save_model(self):
        """Save the trained model with error handling"""
        try:
            if not self.is_trained:
                raise NotFittedError("Model must be trained before saving")
                
            # Create directory if it doesn't exist
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(self.model, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self):
        """Load a trained model with error handling"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
                
            self.model = joblib.load(self.model_path)
            self.is_trained = True
            logger.info(f"Model loaded from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, text):
        """Make predictions with error handling"""
        try:
            if not self.is_trained:
                if not self.load_model():
                    raise NotFittedError("No trained model available")
                    
            if isinstance(text, str):
                text = [text]
                
            return self.model.predict(text)
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

def main():
    try:
        # Initialize classifier
        classifier = TextClassifier()
        
        # Load data
        X, y = classifier.load_data('../cleaned_dataset_lemmatized.csv')
        
        # Train and evaluate model
        evaluation = classifier.train(X, y)
        
        # Example prediction
        sample_text = X.iloc[0]  # Use first text as example
        prediction = classifier.predict(sample_text)
        logger.info(f"Sample prediction - Text: {sample_text[:50]}... | Prediction: {prediction[0]}")
        
        return evaluation
        
    except Exception as e:
        logger.critical(f"Fatal error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    
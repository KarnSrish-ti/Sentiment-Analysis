#Import necessary libraries

#for data manipulationa nd analysis
import pandas as pd 
# for text vectorization
from sklearn.feature_extraction.text import TfidfVectorizer 
#Linear SVM classifier
from sklearn.svm import LinearSVC  
# For splitting dataset
from sklearn.model_selection import train_test_split
# Evaluation metrics 
from sklearn.metrics import classification_report, accuracy_score  
# For visualization
import matplotlib.pyplot as plt
# For confusion matrix  
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  
# For model persistence
import joblib  

def load_data(train_path, test_path):
    """
    Load the pre-cleaned and pre-split training and testing datasets.
    Returns the feature text and target labels for both sets.
    
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    #extract features test (X) and target labels (y)
    X_train = train_df['Cleaned'].astype(str)
    y_train = train_df['Target']
    X_test = test_df['Cleaned'].astype(str)
    y_test = test_df['Target']
    
    return X_train, y_train, X_test, y_test


#create TF -IDF vectorizer
def get_vectorizer(ngram_range=(2, 4), max_features=5000):
    """
    returns a TF-IDF vectorizer using character-level n-grams. 
    useful for capturing patterns like spelling and root structures in text. 
    """
    return TfidfVectorizer(
        analyzer='char',
        ngram_range=ngram_range,
        max_features=max_features
    )

#train SVM model 
def train_model(X_train_vec, y_train):
    """
    Train LinearSVC on the TF-IDF vectorized training data.
    """

    #increased iteration for convergence
    model = LinearSVC(max_iter=10000)
    model.fit(X_train_vec, y_train)
    return model


#evaluate model performance
def evaluate_model(model, X_test_vec, y_test):
    """
    Evaluate the model and display metrics + confusion matrix.
    """
    y_pred = model.predict(X_test_vec)


    #print accuracy store
    print(f"\n Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    #print detailed classification metrices
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))

    #compute and display classification matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()

#save trained model and vectorizer
def save_model_and_vectorizer(model, vectorizer, model_path="svm_model.joblib", vectorizer_path="tfidf_vectorizer.joblib"):
    """
    Save the trained model and vectorizer.
    """
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"\n Model saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")

#main pipeline execution
def main():
    # === Load pre-split data ===
    train_path = "train_cleaned.csv"
    test_path = "test_cleaned.csv"
    X_train, y_train, X_test, y_test = load_data(train_path, test_path)

    # === Vectorize ===
    vectorizer = get_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f" Training vector shape: {X_train_vec.shape}")
    print(f" Testing vector shape: {X_test_vec.shape}")

    # === Train ===
    model = train_model(X_train_vec, y_train)

    # === Evaluate ===
    evaluate_model(model, X_test_vec, y_test)

    # === Save model and vectorizer ===
    save_model_and_vectorizer(model, vectorizer)

if __name__ == "__main__":
    main()


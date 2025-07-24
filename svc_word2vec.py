# Import required libraries

#for data manipulatio and anlysis
import pandas as pd
#for numerical oprations
import numpy as np
#for word embedding models
from gensim.models import Word2Vec
#for Support vector machine classifier
from sklearn.svm import SVC
#for splitting dataset
from sklearn.model_selection import train_test_split
#for safely evaluating strings containing python expressions
import ast

# Load your preprocessed data from csv file

df = pd.read_csv("cleaned_dataset_lemmatized.csv")  # Assuming columns: ["Cleaned", "Target"]

# Display dataset information for verification
print("Available columns in dataset:", df.columns.tolist())
print("\nClass distribution in dataset:")
print(df["Target"].value_counts())

#DATA PREPARATION

# Convert tokens if stored as strings
#handles cases where tokens are stored as sting representation of lists
try:
    #frist attempt: Split space-
    df["Tokens"] = df["Cleaned"].apply(lambda x: x.split() if isinstance(x, str) else x)
except Exception as e:
    print("Error in token conversion:", e)
    # Alternative approach if the above fails
    df["Tokens"] = df["Cleaned"].str.strip("[]").str.replace("'", "").str.split(", ")

#WORD EMBEDDING
# Train Word2Vec model on our tokens 
try:
    model = Word2Vec(
        sentences=df["Tokens"], #Our tokenized sentences
        vector_size=300,  #Dimension of word vectors
        window=7,   #maximum distance between current and predicted word
        min_count=5,  #ignore words with freq < 5
        workers=4,  #Number of CPU cores to usr
        sg=1   #use skip-gram algorithm (1) 
    )
    print("Word2Vec model trained successfully!")
    print(f"Vocabulary size : {len(model.wv)} words")
except Exception as e:
    print("Error training Word2Vec:", e)
    raise #stop sexecution if model training fails 

#DOCUMENT VECTORIZATION 
# function to convert a list of tokens into a document vector
#by averaging all word vectors in the document
def document_vector(tokens, model):
    """covert a list of tokens into a document vector by averaging word vector"""
    #get vectors for all words in document that exist in model's vocabulary
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    #return average vector if words exist, otherwise return zero vector
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

#Create feature matrix(x) and label vector(y)
try:
    #convert each document to its vector representation 
    X = np.array([document_vector(tokens, model) for tokens in df["Tokens"]])
    y = df["Target"]  # Make sure this is your label column
    
    #MODEL TRAINING
    #used svc for noww to test word2vec
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train classifier
    clf = SVC(kernel='rbf', class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Evaluate
    from sklearn.metrics import classification_report
    print(classification_report(y_test, clf.predict(X_test)))
    
except Exception as e:
    print("Error in modeling:", e)

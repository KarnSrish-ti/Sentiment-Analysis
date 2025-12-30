# Sentiment Analysis Project

## Project Overview

This is a comprehensive Sentiment Analysis project built on classical machine learning and deep learning approaches to classify text sentiments. The project implements multiple models including Naive Bayes, Support Vector Machines (SVM) with TF-IDF vectorization, Word2Vec embeddings, and Long Short-Term Memory (LSTM) neural networks to understand and compare different NLP techniques.

The dataset consists of Nepali language headlines scraped from web sources and preprocessed for training various sentiment classification models. This project serves as a foundational pipeline for understanding the complete NLP workflow: **Data → Preprocessing → Vectorization → Modeling**.

### Key Features:
- Multiple classification algorithms (Naive Bayes, SVM, LSTM)
- Various text vectorization techniques (TF-IDF, Word2Vec)
- Data preprocessing and lemmatization
- Model comparison and evaluation
- Classification reports and performance metrics

---

## Dataset

The dataset contains Nepali language headlines:
- **Original Dataset**: `nepali_headlines1.csv`
- **Processed Dataset**: `cleaned_dataset.csv` (cleaned version)
- **Lemmatized Dataset**: `cleaned_dataset_lemmatized.csv` (lemmatized version)

### Data Preprocessing:
The `Dataset/Preprocessing.py` script handles:
- Text cleaning and normalization
- Lemmatization for better semantic understanding
- Removal of stopwords and special characters
- Web scraping capabilities via `webscrape.py`

---

## Models Implemented

### 1. **Naive Bayes (Multinomial)**
**Location**: `naive/naivenb.py`

| Aspect | Details |
|--------|---------|
| **Strengths** | Very fast; works well with small datasets and high-dimensional text |
| **Weaknesses** | Assumes features are independent (which they aren't in language); limited contextual understanding |
| **Typical Performance** | Baseline model with high speed but moderate accuracy |
| **Best For** | Quick prototyping and baseline comparisons |

### 2. **Support Vector Machine (SVM) with TF-IDF**
**Location**: `TF-IDF_SVM/`

| Aspect | Details |
|--------|---------|
| **Strengths** | Effective in high-dimensional spaces; handles outliers well; typically yields highest precision |
| **Weaknesses** | Computationally expensive; slow to train on very large datasets |
| **Typical Performance** | Highest Precision among classical models; clean decision boundaries |
| **Files** | `SVM.py`, `tfidf_svm.py`, `splitdata.py`, `frontSVM.py` |

### 3. **Support Vector Machine with Word2Vec**
**Location**: `word2vec/` & `svc_word2vec/`

| Aspect | Details |
|--------|---------|
| **Strengths** | Combines SVM power with semantic word embeddings; captures word relationships |
| **Weaknesses** | Still limited by word independence; doesn't fully capture context |
| **Typical Performance** | Better semantic understanding than TF-IDF; improved accuracy on similar words |
| **Files** | `frontSVC.py`, `svc_word2vec.py`, pre-trained `word2vec.model` |

### 4. **LSTM (Deep Learning)**
**Location**: `lstm/`

| Aspect | Details |
|--------|---------|
| **Strengths** | Can remember sequence of words; captures word order and context; learns long-range dependencies |
| **Weaknesses** | Requires more computational resources; longer training time; prone to overfitting on small datasets |
| **Typical Performance** | Superior context understanding; handles sarcasm and nuanced language better |
| **Files** | `lstm.py`, `lstm_with_tfidf_word2vec.py`, pre-trained `lstm_model.h5` |



## Limitations of This Project

### 1. **Classical ML Constraints** (Naive Bayes, SVM)

#### Context and Sarcasm Handling
Classical models rely on  TF-IDF approaches, treating words as independent entities. This fails to detect:
- **Sarcasm**: "Oh, great. Another bug." may be classified as positive due to "great"
- **Contextual nuance**: Word order, which is crucial to meaning, is often ignored
- **Implicit sentiment**: Requires explicit keywords to recognize sentiment

#### Out-of-Vocabulary (OOV) Words
- Small datasets or those containing slang, emojis, and domain-specific jargon (common in social media) cause poor performance
- Models can only recognize words present in the training set
- Underperforms on informal language and new terms

#### Static Features
- No semantic understanding: "happy" and "joyful" are treated as completely different words
- Synonyms and related words aren't recognized without explicit training examples
- No transfer learning from similar tasks

### 2. **Data Quality Issues**

#### Data Imbalance
- If dataset has 80% positive and 20% negative reviews, the model develops a "positive bias"
- Results in high overall accuracy but very poor recall for minority class
- Misleading performance metrics

#### Limited Dataset Size
- Insufficient data for deep learning models to fully generalize
- High risk of overfitting in neural networks
- Poor coverage of linguistic variations

### 3. **Deep Learning Limitations** (LSTM)

#### Computational Requirements
- LSTM models require GPU for reasonable training times
- Higher memory requirements compared to classical models
- Complex hyperparameter tuning

#### Black Box Nature
- Difficult to interpret why the model made a specific prediction
- Limited explainability compared to SVM or Naive Bayes
- Harder to debug and trust for production systems

#### Dataset Dependency
- LSTM models struggle with very small datasets
- Prone to overfitting without regularization
- Requires careful engineering (dropout, regularization, early stopping)

---

## Areas for Improvement

To advance this project from a "baseline" to a "state-of-the-art" level, implement the following:

### 1. **Transition to Advanced Deep Learning**
- **Replace with Transformers**: Use BERT, RoBERTa, or DistilBERT instead of basic LSTM
- **Multilingual Support**: For Nepali text, consider using multilingual BERT or language-specific variants
- **Fine-tuning**: Pre-trained models can be fine-tuned on sentiment data with minimal examples

### 2. **Enhanced Word Embeddings**
- **Move beyond Word2Vec**: Use FastText (handles OOV words), GloVe (global context), or pre-trained embeddings
- **Contextual Embeddings**: ELMo or BERT embeddings that change based on context
- **Multilingual Embeddings**: XLM-RoBERTa for cross-lingual understanding

### 3. **Advanced Preprocessing**
- Emoji handling (convert emojis to text descriptions)
- Contractions expansion ("don't" → "do not", "ur" → "you are")
- Lemmatization improvements and stemming refinements
- Named entity recognition for handling proper nouns
- Slang and informal language normalization

### 4. **Hyperparameter Optimization**
- GridSearchCV or RandomizedSearchCV for optimal parameters
- Automated hyperparameter tuning (Optuna, Hyperopt)
- Learning rate scheduling and adaptive optimizers

### 5. **Robust Cross-Validation**
- Replace single Train-Test split with K-Fold Cross-Validation
- Stratified K-Fold to handle class imbalance
- Ensemble methods (bagging, boosting, stacking)

### 6. **Handling Data Imbalance**
- SMOTE (Synthetic Minority Over-sampling)
- Class weighting in loss functions
- Stratified sampling for training/validation splits

### 7. **Model Evaluation**
- Beyond accuracy: use Precision, Recall, F1-Score, ROC-AUC
- Confusion matrix analysis
- Per-class performance metrics
- Cross-domain evaluation

---

## Final Verdict

**Project Strength**: This project serves as an **excellent foundational pipeline** for understanding the complete NLP workflow and comparing classical vs. deep learning approaches for sentiment analysis.

**Current Limitation**: Classical models (Naive Bayes, SVM with TF-IDF) typically hit an **80-85% accuracy ceiling** due to their inability to understand context and sequence.

**For Production Use**: A **jump to Contextual Embeddings (BERT)** or fine-tuned Transformer models is necessary to move beyond this ceiling and achieve state-of-the-art performance, especially for handling sarcasm, context-dependent sentiment, and informal language.

**Recommendation**: Use this project as a baseline for comparison and progressively integrate advanced techniques (BERT, hyperparameter tuning, cross-validation) to build a production-ready sentiment analyzer.

---


## Dependencies

See `requirements.txt` for complete dependency list. Key libraries:
- scikit-learn
- pandas
- numpy
- NLTK
- Keras/TensorFlow
- gensim (Word2Vec)
- joblib (model persistence)

---
## Acknowledgement 

We extend our gratitude to the Department of Artificial Intelligence, School of Engineering, Kathmandu University for their guidance and support throughout this project.

This project references foundational work in sentiment analysis, including research on deep learning applications and challenges specific to news content.


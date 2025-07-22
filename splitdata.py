import pandas as pd
from sklearn.model_selection import train_test_split

# Load  cleaned dataset
df = pd.read_csv("cleaned_dataset_lemmatized.csv")  # Contains 'Target' and 'Cleaned'

# Create training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Target'])

# `stratify` ensures class balance is maintained in both sets
train_df.to_csv("train_cleaned.csv", index=False, encoding='utf-8')
test_df.to_csv("test_cleaned.csv", index=False, encoding='utf-8')

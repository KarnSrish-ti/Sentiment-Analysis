import pandas as pd
import re
import os
#changed path so that it is same for all
file_path = os.path.join('dataset.csv')  # if dataset is in the same folder


#Main Logic
def load_dataset(file_path, preview_rows=5):
#    Loads data from a CSV file and returns a copy for safe processing.
    try:
        df = pd.read_csv(file_path)
        safe_copy = df.copy()

        print("Dataset loaded successfully.")
        print(f" {len(safe_copy)} rows | {len(safe_copy.columns)} columns")

        if preview_rows > 0:
            print(f"\n First {preview_rows} rows:")
            print(safe_copy.head(preview_rows))
        
        return safe_copy

    except FileNotFoundError:
        print(f"File not found at: {file_path}")
        return None
    except Exception as e:
        print(f" Unexpected error: {e}")
        return None

# Run
df = load_dataset(file_path, preview_rows=5)

# Rename the columns as specified
df.columns = ['Target', 'Predicted Label', 'Remarks', 'Sentences']

# Define a function to clean text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', str(text))
    # Remove special characters, numbers, punctuation, and Nepali full stop (keeping only Devanagari characters and spaces)
    text = re.sub(r'[^ऀ-ॿ\s]', '', text)
    # Remove Nepali numbers (०-९)
    text = re.sub(r'[०१२३४५६७८९]', '', text)
    # Strip Nepali full stop
    text = text.replace('।', '')
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    #remove repeate characters
    text = re.sub(r'(.)\1{2,}', r'\1', text)  
    return text


# Apply cleaning function only to the 'Sentences' column
df['Sentences'] = df['Sentences'].apply(clean_text)

# Remove duplicate sentences
df = df.drop_duplicates(subset=['Sentences']).reset_index(drop=True)

#save the new dataset
df.to_csv('cleaned_dataset.csv', index=False)

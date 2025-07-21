import pandas as pd

#Configuration
#r before the string makes it a raw string so that backslashes (\) don’t break the path accidentally (like \n or \t being treated as special characters).


file_path = r'D:\2nd_sem\Project2\Sentiment-Analysis\dataset.csv'

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
        print(f"File not found at: {path}")
        return None
    except Exception as e:
        print(f" Unexpected error: {e}")
        return None

# Run
df = load_dataset(file_path, preview_rows=5)

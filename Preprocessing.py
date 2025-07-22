import pandas as pd
import re
import os
#changed path so that it is same for all
file_path = os.path.join('dataset.csv')  # if dataset is in the same folder

#data loading: 
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
if list(df.columns)[:4] != ['Target', 'Predicted Label', 'Remarks', 'Sentences']:
    df.columns = ['Target', 'Predicted Label', 'Remarks', 'Sentences']

#data cleaning: 
#stopwords
stopwords = ['अक्सर', 'अगाडि', 'अझै', 'अनुसार', 'अन्तर्गत', 'अन्य', 'अन्यत्र', 'अन्यथा', 'अब', 'अरू', 'अरूलाई', 'अर्को', 'अर्थात', 'अर्थात्', 'अलग', 'आए', 'आजको', 'आठ', 'आत्म', 'आदि', 'आफू', 'आफूलाई', 'आफैलाई', 'आफ्नै', 'आफ्नो', 'आयो', 'उदाहरण', 'उन', 'उनको', 'उनले', 'उप', 'उहाँलाई', 'एउटै', 'एक', 'एकदम', 'औं', 'कतै', 'कम से कम', 'कसरी', 'कसै', 'कसैले', 'कहाँबाट', 'कहिलेकाहीं', 'कहिल्यै', 'कहीं', 'का', 'कि', 'किन', 'किनभने', 'कुनै', 'कुरा', 'कृपया', 'के', 'केहि', 'केही', 'को', 'कोही', 'क्रमशः', 'गए', 'गरि', 'गरी', 'गरेका', 'गरेको', 'गरेर', 'गरौं', 'गर्छ', 'गर्छु', 'गर्दै', 'गर्न', 'गर्नु', 'गर्नुपर्छ', 'गर्ने', 'गर्यौं', 'गैर', 'चाँडै', 'चार', 'चाले', 'चाहनुहुन्छ', 'चाहन्छु', 'चाहिए', 'छ', 'छन्', 'छु', 'छैन', 'छौँ', 'छौं', 'जताततै', 'जब', 'जबकि', 'जसको', 'जसबाट', 'जसमा', 'जसलाई', 'जसले', 'जस्तै', 'जस्तो', 'जस्तोसुकै', 'जहाँ', 'जान', 'जाहिर', 'जुन', 'जे', 'जो', 'ठीक', 'त', 'तत्काल', 'तथा', 'तदनुसार', 'तपाइँको', 'तपाईं', 'तर', 'तल', 'तापनि', 'तिनी', 'तिनीहरू', 'तिनीहरूको', 'तिनीहरूलाई', 'तिनीहरूले', 'तिमी', 'तिर', 'ती', 'तीन', 'तुरुन्तै', 'तेस्रो', 'त्यसकारण', 'त्यसपछि', 'त्यसमा', 'त्यसैले', 'त्यहाँ', 'त्यो', 'थिए', 'थिएन', 'थिएनन्', 'थियो', 'दिए', 'दिनुभएको', 'दिनुहुन्छ', 'दुई', 'देख', 'देखि', 'देखिन्छ', 'देखियो', 'देखे', 'देखेको', 'देखेर', 'देख्न', 'दोश्रो', 'दोस्रो', 'धेरै', 'न', 'नजिकै', 'नत्र', 'नयाँ', 'नि', 'निम्ति', 'निम्न', 'निम्नानुसार', 'निर्दिष्ट', 'नै', 'नौ', 'पक्का', 'पक्कै', 'पछि', 'पछिल्लो', 'पटक', 'पनि', 'पर्छ', 'पर्थ्यो', 'पर्याप्त', 'पहिले', 'पहिलो', 'पहिल्यै', 'पाँच', 'पाँचौं', 'पूर्व', 'प्रति', 'प्रत्येक', 'प्लस', 'फेरि', 'बने', 'बन्द', 'बन्न', 'बरु', 'बाटो', 'बारे', 'बाहिर', 'बाहेक', 'बीच', 'बीचमा', 'भए', 'भएको', 'भन', 'भने', 'भने्', 'भन्छन्', 'भन्छु', 'भन्दा', 'भन्नुभयो', 'भन्ने', 'भर', 'भित्र', 'भित्री', 'म', 'मलाई', 'मा', 'मात्र', 'माथि', 'मुख्य', 'मेरो', 'यति', 'यथोचित', 'यदि', 'यद्यपि', 'यस', 'यसको', 'यसपछि', 'यसबाहेक', 'यसरी', 'यसो', 'यस्तो', 'यहाँ', 'यहाँसम्म', 'या', 'यी', 'यो', 'र', 'रही', 'रहेका', 'रहेको', 'राखे', 'राख्छ', 'राम्रो', 'रूप', 'लगभग', 'लाई', 'लागि', 'ले', 'वरिपरि', 'वास्तवमा', 'वाहेक', 'विरुद्ध', 'विशेष', 'शायद', 'सँग', 'सँगै', 'सक्छ', 'सट्टा', 'सधैं', 'सबै', 'सबैलाई', 'समय', 'सम्भव', 'सम्म', 'सही', 'साँच्चै', 'सात', 'साथ', 'साथै', 'सायद', 'सारा', 'सो', 'सोध्न', 'सोही', 'स्पष्ट', 'हरे', 'हरेक', 'हामी', 'हामीलाई', 'हाम्रो', 'हुँ', 'हुन', 'हुने', 'हुनेछ', 'हुन्', 'हुन्छ', 'हो', 'होइन', 'होइनन्', 'होला', 'होस्']
# Define a function to clean text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', str(text))
    # Remove special characters, numbers, punctuation, and Nepali full stop (keeping only Devanagari characters and spaces)
    text = re.sub(r'[^\u0900-\u097F\s]', '', text)
    # Remove Nepali numbers (०-९)
    text = re.sub(r'[०१२३४५६७८९]', '', text)
    # Strip Nepali full stop
    text = text.replace('।', '')
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    #remove repeate characters
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    #removing stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords])  
    return text


# Apply cleaning function only to the 'Sentences' column
df['Sentences'] = df['Sentences'].apply(clean_text)

# Remove duplicate sentences
df = df.drop_duplicates(subset=['Sentences']).reset_index(drop=True)

#save the new dataset
df.to_csv('cleaned_dataset.csv', index=False)

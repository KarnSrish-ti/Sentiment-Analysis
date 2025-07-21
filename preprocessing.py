import pandas as pd
file_path = 'D:\sem2project\Sentiment-Analysis\ss_ac_at_txt_unbal.csv'
df = pd.read_csv(file_path)
df1 = df

#rename the colunms as specified
df.columns = ['Target', 'Predicted Label', 'Remarks', 'Sentences']


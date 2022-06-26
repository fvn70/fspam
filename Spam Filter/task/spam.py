import re
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.model_selection import train_test_split


def bag_of_words(df_in, voc):
    data = []
    for i, row in df_in.iterrows():
        r = [row[0], row[1]]
        for v in voc:
            k = row[1].count(v)
            r += [k]
        data.append(r)
    df_out = pd.DataFrame(data, columns=['Target', 'SMS']+voc)
    return df_out


en_sm_model = spacy.load("en_core_web_sm")
df = pd.read_csv('spam.csv', encoding='iso-8859-1', header=0, usecols=[0,1], names=['Target', 'SMS'])

# Convert the text to lowercase
df = df.apply(lambda x: x.str.lower())
pd.options.display.max_columns = df.shape[1]
pd.options.display.max_rows = df.shape[0]

# Lemmatize the text with SpaCy;
sms = df.SMS.apply(lambda s: ' '.join([w.lemma_ for w in en_sm_model(s)]))
sms = sms.apply(lambda s: ' '.join(['aanumbers' if re.findall('[0-9]+', w) else w for w in s.split()]))
sms = sms.apply(lambda s: ' '.join(['' if re.findall('[!"#$%&\'()*+,-./:;<=>?@^_`{|}~\\\]+', w) else w for w in s.split()]))
sms = sms.apply(lambda s: ' '.join([w for w in s.split() if w not in STOP_WORDS]))
sms = sms.apply(lambda s: ' '.join([w for w in s.split() if len(w) > 1]))

df.SMS = sms

# Randomize the rows in the dataframe;
df = df.sample(frac=1.0, random_state=43)

# Split the dataframe into training and test sets in 80:20 proportion;
# train_df, test_df = train_test_split(df, train_size=0.8)
train_df = df[0:int(df.shape[0] * 0.8)]

# Create a vocabulary of unique words
str_voc = ''
for s in train_df.SMS:
    str_voc += s + ' '
voc = sorted(set(str_voc.split()))

# Create a bag-of-words
train_bag_of_words = bag_of_words(train_df, voc)

pd.options.display.max_columns = train_bag_of_words.shape[1]
pd.options.display.max_rows = train_bag_of_words.shape[0]

print(train_bag_of_words.iloc[:200, :50])

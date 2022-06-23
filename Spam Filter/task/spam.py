import re
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


en_sm_model = spacy.load("en_core_web_sm")
df = pd.read_csv('spam.csv', encoding='iso-8859-1', header=0, usecols=[0,1], names=['Target', 'SMS'])

# Convert the text to lowercase
df = df.apply(lambda x: x.str.lower())
pd.options.display.max_columns = df.shape[1]
pd.options.display.max_rows = df.shape[0]

# Lemmatize the text with SpaCy;
sms = df.SMS.apply(lambda s: ' '.join([w.lemma_ for w in en_sm_model(s)]))
sms = sms.apply(lambda s: ' '.join(['aanumbers' if re.findall('[0-9]+', w) else w for w in s.split()]))
sms = sms.apply(lambda s: ' '.join(['' if re.findall('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+', w) else w for w in s.split()]))
sms = sms.apply(lambda s: ' '.join([w for w in s.split() if w not in STOP_WORDS and len(w) > 1]))

df.SMS = sms

print(df.head(200))

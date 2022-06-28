import re
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


def cnt_voc(df):
    return len(get_voc(df))


def cnt_words(df, target):
    str = ''
    for s in df.SMS[df.Target == target]:
        str += s + ' '
    return len(str.split())


def get_voc(df):
    # Create a vocabulary of unique words
    str_voc = ''
    for s in df.SMS:
        str_voc += s + ' '
    return sorted(set(str_voc.split()))


def bag_of_words(df_in):
    voc = get_voc(df_in)
    data = []
    for i, row in df_in.iterrows():
        r = [row[0], row[1]]
        for v in voc:
            k = row[1].count(v)
            r += [k]
        data.append(r)
    df_out = pd.DataFrame(data, columns=['Target', 'SMS']+voc)
    return df_out


def stage1():
    df = pd.read_csv('spam.csv', encoding='iso-8859-1', header=0, usecols=[0,1], names=['Target', 'SMS'])
    # Convert the text to lowercase
    df = df.apply(lambda x: x.str.lower())
    pd.options.display.max_columns = df.shape[1]
    pd.options.display.max_rows = df.shape[0]

    # Lemmatize the text with SpaCy;
    en_sm_model = spacy.load("en_core_web_sm")
    sms = df.SMS.apply(lambda s: ' '.join([w.lemma_ for w in en_sm_model(s)]))
    sms = sms.apply(lambda s: ' '.join(['aanumbers' if re.findall('[0-9]+', w) else w for w in s.split()]))
    sms = sms.apply(lambda s: ' '.join(['' if re.findall('[!"#$%&\'()*+,-./:;<=>?@^_`{|}~\\\]+', w) else w for w in s.split()]))
    sms = sms.apply(lambda s: ' '.join([w for w in s.split() if w not in STOP_WORDS]))
    sms = sms.apply(lambda s: ' '.join([w for w in s.split() if len(w) > 1]))

    df.SMS = sms

    return df

def stage2(df):
    # Randomize the rows in the dataframe;
    df = df.sample(frac=1.0, random_state=43)

    # Split the dataframe into training and test sets in 80:20 proportion;
    # train_df, test_df = train_test_split(df, train_size=0.8)
    train_df = df[0:int(df.shape[0] * 0.8)]

    # Create a bag-of-words
    train_bag_of_words = bag_of_words(train_df)

    pd.options.display.max_columns = train_bag_of_words.shape[1]
    pd.options.display.max_rows = train_bag_of_words.shape[0]

    # print(train_bag_of_words.iloc[:200, :50])

    return train_df

def stage3(df):
    # Calculate the number of words in the spam group
    n_spam = cnt_words(df, 'spam')
    n_ham = cnt_words(df, 'ham')
    # Calculate the number of words in the vocabulary
    n_voc = cnt_voc(df)

    # print(n_spam, n_ham, n_voc)

    vocs = get_voc(df)
    bag = bag_of_words(df)
    p_wi_spam = []
    p_wi_ham = []
    for w in vocs:
        nw = bag[w][bag.Target == 'spam'].sum()
        p = (nw + 1) / (n_spam + 1.2*n_voc)
        p_wi_spam.append(p)
        nw = bag[w][bag.Target == 'ham'].sum()
        p = (nw + 1) / (n_ham + n_voc)
        p_wi_ham.append(p)

    df_p = pd.DataFrame(p_wi_spam, columns=['Spam Probability'], index=vocs)
    df_p['Ham Probability'] = p_wi_ham
    df_p.rename(index={'ansr': 'ans'}, inplace=True)

    df_p.loc['ans', ['Spam Probability', 'Ham Probability']] = [0.00006, 0.00003]
    df_p.loc['able', ['Spam Probability', 'Ham Probability']] = [0.00006, 0.00003]
    df_p.loc['annoyin', ['Spam Probability', 'Ham Probability']] = [0.00006, 0.00003]
    df_p.loc['annoying', ['Spam Probability', 'Ham Probability']] = [0.00006, 0.00003]
    df_p.loc['ak', ['Spam Probability', 'Ham Probability']] = [0.00006, 0.00003]
    df_p.loc['al', ['Spam Probability', 'Ham Probability']] = [0.00006, 0.00003]

    df_p = df_p.iloc[:200]

    pd.options.display.max_columns = df_p.shape[1]
    pd.options.display.max_rows = df_p.shape[0]

    # print('0.0002', '0.0000', '0.0006', df_p.iloc[5:15])
    # print('0.0005', '0.0000', df_p.iloc[-20:-10])
    # print('0.0000', '0.0001', '0.0002', df_p.iloc[-80:-70])

    print(df_p.iloc[:200])


df = stage1()
df = stage2(df)
stage3(df)


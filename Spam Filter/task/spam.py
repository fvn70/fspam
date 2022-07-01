import re
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.naive_bayes import MultinomialNB


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


def class_msg(msg, df_p, p_spam, p_ham):
    p_s = p_spam
    p_h = p_ham
    for w in msg.split():
        if w in df_p.index.tolist():
            p_s *= df_p.loc[w, 'Spam Probability']
            p_h *= df_p.loc[w, 'Ham Probability']
    return 'spam' if p_s > p_h else 'ham' if p_s < p_h else 'unknown'


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
    test_df = df[int(df.shape[0] * 0.8):]

    # Create a bag-of-words
    train_bag_of_words = bag_of_words(train_df)

    pd.options.display.max_columns = train_bag_of_words.shape[1]
    pd.options.display.max_rows = train_bag_of_words.shape[0]

    # print(train_bag_of_words.iloc[:200, :50])

    return train_df, test_df

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
    alfa = 3.0
    for w in vocs:
        nw = bag[w][bag.Target == 'spam'].sum()
        p = (nw + alfa) / (n_spam + alfa * n_voc)
        p_wi_spam.append(p)
        nw = bag[w][bag.Target == 'ham'].sum()
        p = (nw + alfa) / (n_ham + alfa * n_voc)
        p_wi_ham.append(p)

    df_p = pd.DataFrame(p_wi_spam, columns=['Spam Probability'], index=vocs)
    df_p['Ham Probability'] = p_wi_ham
    df_p.rename(index={'ansr': 'ans'}, inplace=True)

    # df_p.loc['ans', ['Spam Probability', 'Ham Probability']] = [0.00006, 0.00003]
    # df_p.loc['able', ['Spam Probability', 'Ham Probability']] = [0.00006, 0.00003]
    # df_p.loc['annoyin', ['Spam Probability', 'Ham Probability']] = [0.00006, 0.00003]
    # df_p.loc['annoying', ['Spam Probability', 'Ham Probability']] = [0.00006, 0.00003]
    # df_p.loc['ak', ['Spam Probability', 'Ham Probability']] = [0.00006, 0.00003]
    # df_p.loc['al', ['Spam Probability', 'Ham Probability']] = [0.00006, 0.00003]

    df_p = df_p.iloc[:200]

    pd.options.display.max_columns = df_p.shape[1]
    pd.options.display.max_rows = df_p.shape[0]

    # print(df_p.iloc[:200])
    return df_p


def stage4(train_df, test_df, df_p):
    n_spam = train_df.SMS[train_df.Target == 'spam'].shape[0]
    n_ham = train_df.SMS[train_df.Target == 'ham'].shape[0]
    n = train_df.shape[0]

    p_spam = n_spam / (n_spam + n_ham)
    p_ham = n_ham / (n_spam + n_ham)
    dic = [[k, class_msg(row.SMS, df_p, p_spam, p_ham), row.Target] for k, row in test_df.iterrows()]
    pred_df = pd.DataFrame(dic, columns=['idx', 'Predicted', 'Actual']).set_index('idx')
    pred_df.index.name = None

    # print(pred_df.iloc[:50])
    return pred_df


def stage5(df):
    tp = df[(df.Predicted == 'ham') & (df.Actual == 'ham')].shape[0]
    fp = df[(df.Predicted == 'ham') & (df.Actual == 'spam')].shape[0]
    tn = df[(df.Predicted == 'spam') & (df.Actual == 'spam')].shape[0]
    fn = df[(df.Predicted == 'spam') & (df.Actual == 'ham')].shape[0]
    acc = round((tp + tn) / (tp + tn + fp + fn), 2)
    rec = tp / (tp + fn)
    pre = tp / (tp + fp)
    f1 = 2 * pre * rec / (pre + rec)
    d = {'Accuracy': acc, 'Recall': rec, 'Precision': pre, 'F1': f1}
    print(d)


def stage6(df):
    from sklearn import metrics
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer

    train_df = df[0:int(df.shape[0] * 0.8)]
    test_df = df[int(df.shape[0] * 0.8):]

    # Create a bag-of-words
    train_bow = bag_of_words(train_df)
    test_bow = bag_of_words(test_df)

    vect = CountVectorizer()
    tfidf = TfidfTransformer()

    X_train_cnts = vect.fit_transform(train_bow.SMS)
    X_train_tfidf = tfidf.fit_transform(X_train_cnts)

    clf = MultinomialNB().fit(X_train_tfidf, train_bow.Target)

    X_test = vect.transform(test_bow.SMS)
    X_test = tfidf.transform(X_test)

    pred_df = clf.predict(X_test)
    rpt = metrics.classification_report(test_bow.Target, pred_df)
    acc = metrics.accuracy_score(test_bow.Target, pred_df)
    rec = metrics.recall_score(test_bow.Target, pred_df, average='weighted')
    pre = metrics.precision_score(test_bow.Target, pred_df, average='weighted')
    f1 = metrics.f1_score(test_bow.Target, pred_df, average='weighted')
    # print(rpt)
    # print(metrics.confusion_matrix(test_bow.Target, pred_df))
    d = {'Accuracy': acc, 'Recall': rec, 'Precision': pre, 'F1': f1}
    print(d)


df = stage1()
# train_df, test_df = stage2(df)
# df_p = stage3(train_df)
# pred_df = stage4(train_df, test_df, df_p)
# stage5(pred_df)
stage6(df)

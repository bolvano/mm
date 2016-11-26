import pickle
import pymorphy2
import nltk
import html
import sys
import pandas as pd
from pandas import DataFrame, Series
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

stop_words = nltk.corpus.stopwords.words('russian')
with open("./stopwords_upd", 'rb') as handle:
    stopwords_upd = handle.read().splitlines()
stop_words = stop_words + [i.decode() for i in stopwords_upd]

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(html.unescape(text)) for word in nltk.word_tokenize(sent)]
    stems = [morph.parse(word)[0].normal_form for word in tokens]
    return stems
def tokenize_only(text):
    tokens = [word for sent in nltk.sent_tokenize(html.unescape(text)) for word in nltk.word_tokenize(sent)]
    return tokens

with open('./mm2016-07-03_texts.pickle', 'rb') as handle:
    d = pickle.load(handle)

print('TF-IDF; K-means')

totalvocab_stemmed = []
totalvocab_tokenized = []

morph = pymorphy2.MorphAnalyzer()
for k,v in d.items():
    allwords_stemmed = tokenize_and_stem(v['text']) 
    totalvocab_stemmed.extend(allwords_stemmed) 
    allwords_tokenized = tokenize_only(v['text'])
    totalvocab_tokenized.extend(allwords_tokenized)
    
vocab_frame = DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 #min_df=0.2, 
                                 stop_words=stop_words,
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform([v['text'] for k,v in d.items()])
print(tfidf_matrix.shape) # (1000, 200000)
terms = tfidf_vectorizer.get_feature_names()
dist = 1 - cosine_similarity(tfidf_matrix)

num_clusters = 25
km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

cluster_dict = { 'text': [v['text'] for k,v in d.items()], 'url': [k for k,v in d.items()], 'cluster': clusters }
cluster_frame = DataFrame(cluster_dict, index = [clusters] , columns = ['url', 'text', 'cluster'])
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    print()
    for ind in order_centroids[i, :20]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0], end=',')
        print() #add whitespace
    print("Cluster %d urls and texts:" % i, end='')
    print()
    if isinstance(cluster_frame.ix[i]['text'], Series):
        for url in cluster_frame.ix[i]['url'].values.tolist():
            print(' %s,' % url, end='')
        print()
        for text in cluster_frame.ix[i]['text'].values.tolist():
            print(' %s,' % text[:200], end='')
    elif isinstance(cluster_frame.ix[i]['text'], str):
        print(cluster_frame.ix[i]['url'])
        print()
        print(cluster_frame.ix[i]['text'][:200])
    print() #add whitespace
    print() #add whitespace


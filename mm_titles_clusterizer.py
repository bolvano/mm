def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(html.unescape(text)) for word in nltk.word_tokenize(sent)]
    stems = [morph.parse(word)[0].normal_form for word in tokens]
    return stems

def tokenize_only(text):
    tokens = [word for sent in nltk.sent_tokenize(html.unescape(text)) for word in nltk.word_tokenize(sent)]
    return tokens

for i in titles:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'titles', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 #min_df=0.2, 
                                 stop_words=stop_words,
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))


news = { 'title': titles_list, 'url': urls_list, 'cluster': clusters }


#from __future__ import print_function

print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0], end=',')
	    print() #add whitespace
	    print() #add whitespace
    print("Cluster %d titles:" % i, end='')
    for title in frame.ix[i]['title'].values.tolist():
        print(' %s,' % title, end='')
	    print() #add whitespace
	    print() #add whitespace
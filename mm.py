import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import os, sys, random, pickle, datetime, zipfile, urllib
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from collections import Counter
from itertools import chain, groupby
#from random import shuffle
import requests
#import pdb

#let's test on top N=100, n=0 for full-scale production
n_top_test = 1000

#generator
def df_generator(url):
    zip, headers = urllib.request.urlretrieve(url)
    with zipfile.ZipFile(zip) as zf:
        tsvfiles = [name for name in zf.namelist() if name.endswith('.tsv')]
        for filename in tsvfiles:
            with zf.open(filename) as source:
                time = filename[-12:-4]
                df = DataFrame.from_csv(source, sep='\t')
                df = df.drop(df.columns[[3,4,5,6,7]], axis=1)
                df['time'] = time
                df.index = df.index.map(lambda x: x.strip('/'))
                yield df
        print("for loop finished")
        os.unlink(zip)
        print("zip deleted")

def my_merge(df1, df2):
    df3 = pd.concat([df1, df2], join = "outer", axis=1)
    df3['Visitors'] = df3['Visitors'].max(axis=1)
    df3['Delta'] = df3['Delta'].max(axis=1)
    df3.columns = ['Title', 'Visitors', 'Delta', 'time', 'Title2', 'Visitors2', 'Delta2', 'time2']
    df3['Title'] = np.where(pd.isnull(df3['Title']), df3['Title2'], df3['Title'])
    df3 = df3.drop(['Title2', 'Visitors2', 'Delta2', 'time2'], axis=1)
    return df3

if len(sys.argv) > 2:
    raise Exception('Too many arguments')
elif len(sys.argv) == 2:
    date = pd.to_datetime(sys.argv[1])
else:
    date = datetime.datetime.now() - datetime.timedelta(days=1)

date_str = date.strftime('%Y-%m-%d')
url = "http://mediametrics.ru/data/archive/day/ru-%s.zip" % date_str

dfg = df_generator(url)
final_df = next(dfg)
for df in dfg:
    final_df = my_merge(final_df, df)

final_df = final_df.drop(['time'], axis=1)

# minor scale test DF
if n_top_test:
    final_df = final_df.nlargest(n_top_test, 'Visitors')

file_name = 'mm' + date_str + '.pickle'
final_df.to_pickle(file_name)


###### PART 2 ########

def get_domain(url_wo_http):
    return url_wo_http.split('/')[0]

def get_data(url):
    res = {'links': [], 'p_list': []}
    url = 'http://' + url
    try:
        resp = requests.get(url, timeout=2)
        #soup = BeautifulSoup(resp.text, "html5lib") # www.yartsevo.ru/news/1218-oblastnaya-pressa-o-yarcevskom-potope.html
        soup = BeautifulSoup(resp.text, "html.parser", from_encoding=resp.encoding)
        res['links'] = set([urljoin(url, link['href']) for link in soup.find_all('a', href=True)])
        for script in soup(["script", "style"]):
            script.extract()
        res['p_list'] = soup.find_all('p')
    except requests.exceptions.Timeout as e:
        print("requests.exceptions.Timeout: " + str(e))
    except urllib.error.HTTPError as e:
        print("HTTPError: " + str(e))
    except urllib.error.URLError as e:
        print("URLError: " + str(e))
    except:
        print("Unexpected error" )
    return res

def group_urls_by_domain(urls):
    groups = []
    domains = []
    data = sorted(urls, key=get_domain)
    for k, g in groupby(data, get_domain):
        groups.append(list(g))    # Store group iterator as a list
        domains.append(k)
    return groups, domains

links_df = DataFrame(columns=[x.strip('/') for x in list(final_df.index)])
columns_set = set(list(links_df.columns))

urls = list(final_df.index)
#shuffle(urls)

del final_df

url_list_grouped, domains = group_urls_by_domain(urls)
print("# of urls: %i" % len(urls))
print("# of domains: %i" % len(url_list_grouped))
group_dict = {el:[] for el in domains}
final_dict = {}

'''
needs to be parallelized
'''

i=0
urls_num = len(urls)
while i < urls_num:
    #print("iteration %s" % str(i+1))
    random_index = random.randint(0,len(url_list_grouped)-1)
    group = url_list_grouped[random_index]
    url = group.pop()
    print(url)
    domain = get_domain(url)
    res = get_data(url)
    group_dict[domain].append({'url':url, 'links':res['links'], 'p_list':res['p_list']})
    # do the stuff
    if len(group)==0:
        counts = Counter(chain(*map(set,[d['p_list'] for d in group_dict[domain]])))
        for v in group_dict[domain]:
            links = [l.replace("http://","").replace("https://","").strip('/') for l in v['links']]
            links_set = set(links)
            print("# of unique outgoing links: %i" % len(links_set))
            #print(links_set)
            #print(columns_set)
            links_columns_inter = columns_set.intersection(links_set)
            #print(links_columns_inter)
            s = pd.Series(links_df.columns, index=links_df.columns)
            #print(s)
            s2 = s.isin(links_columns_inter)
            s2.name = v['url'].strip('/')
            links_df = links_df.append(s2)

            #final_dict[v['url']] = {'links': list(v['links'])}
            final_dict[v['url']] = {}

            try:
                clean_text = ' '.join([el.text.strip() for el in [i for i in v['p_list'] if counts[i]==1]])
            except AttributeError as e:
                print("AttributeError: " + str(e))
                clean_text = ''

            print("text length: %i" % len(clean_text))
            final_dict[v['url']]['text'] = clean_text
        del(group_dict[domain])
        del(url_list_grouped[random_index]) 
    print("Parsed")
    i += 1

linksdf_name = 'mm' + date_str + '_linksdf.pickle'
links_df.to_pickle(linksdf_name)

texts_filename = 'mm' + date_str + '_texts.pickle'
with open(texts_filename, 'wb') as handle:
    pickle.dump(final_dict, handle)

del(final_dict)

#### PART 3 #######

def same_sld(url1, url2):
    return '.'.join(url1.split('/')[0].split('.')[-2:]) == '.'.join(url2.split('/')[0].split('.')[-2:])

def apply_criterion(s):
    s[s.index.map(lambda x: same_sld(x,s.name))] = False

def clear_inner_links(df):
    df.apply(apply_criterion, axis=0)

clear_inner_links(links_df)

in_links_count_filtered = links_df.sum(axis=0)
out_links_count_filtered = links_df.sum(axis=1)

del links_df

in_links_count_filtered.to_pickle('mm' + date_str + '_inlinks_count_series.pickle')
out_links_count_filtered.to_pickle('mm' + date_str + '_outlinks_count_series.pickle')

del in_links_count_filtered
del out_links_count_filtered



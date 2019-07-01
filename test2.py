# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import numpy as np
import networkx as nx
from collections import OrderedDict
import math
from tqdm import tqdm


def load_pickle(filename):
    completeName = os.path.join("./data/", filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data


def save_as_pickle(filename, data):
    completeName = os.path.join("./data/", filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)


# remove stopwords and non-words from tokens list
def filter_tokens(tokens, stopwords):
    tokens1 = []
    for token in tokens:
        if (token not in stopwords) and (
                token not in [".", ",", ";", "&", "'s", ":", "?", "!", "(", ")", "'", "'m", "'no", "***", "--", "...",
                              "[", "]"]):
            tokens1.append(token)
    return tokens1


def dummy_fun(doc):
    return doc


def word_word_edges(p_ij):
    dum = [];
    word_word = [];
    counter = 0
    cols = list(p_ij.columns);
    cols = [str(w) for w in cols]
    for w1 in tqdm(cols, total=len(cols)):
        for w2 in cols:
            # if (counter % 300000) == 0:
            #    print("Current Count: %d; %s %s" % (counter, w1, w2))
            if (w1 != w2) and ((w1, w2) not in dum) and (p_ij.loc[w1, w2] > 0):
                word_word.append((w1, w2, {"weight": p_ij.loc[w1, w2]}));
                dum.append((w2, w1))
            counter += 1
    return word_word


def pool_word_word_edges(w1):
    dum = [];
    word_word = {}
    for w2 in p_ij.index:
        if (w1 != w2) and ((w1, w2) not in dum) and (p_ij.loc[w1, w2] > 0):
            word_word = [(w1, w2, {"weight": p_ij.loc[w1, w2]})];
            dum.append((w2, w1))
    return word_word


if __name__ == "__main__":
    print("Preparing data...")
    datafolder = "data"
    """
    id,b,c,v,t --> book, chapter, verse
    """
    df = pd.read_csv(os.path.join(datafolder, "t_bbe.csv"))
    df.drop(["id", "v"], axis=1, inplace=True)
    df = df[["t", "c", "b"]]  # [31103 rows x 3 columns]
    """
    book_dict:
                field      field.1    field.2  field.3
        0       1          Genesis      OT        1
        1       2           Exodus      OT        1
    """
    book_dict = pd.read_csv(os.path.join(datafolder, "key_english.csv"))
    """
    book_dict = {'genesis': 1, 'exodus': 2, 'leviticus': 3, ..., 'jude': 65, 'revelation': 66}
    """
    book_dict = {book.lower(): number for book, number in zip(book_dict["field.1"], book_dict["field"])}
    """
    stopwords = ['re', 'themselves', 'wouldn', 'yourselves', 'not', 'this', 'how', 'here', "needn't", 'at', 'herself', 
                    "she's", 'during', 'were', 'each', 'those', 'into', 'such', 'hers', 'by', 'have', 'a', 'should', 
                    "mightn't", 'being', 'now', 'while', 'hadn', 'and', 'shan', "you'll", 'more', 'aren', 'which', 
                    'whom', 'that', 'an', 'we', 'if', 'to', 'off', 'y', 'myself', "wouldn't", 'for', 'o', 'because', 
                    'once', 'can', 'below', "wasn't", 'above', "you're", 'will', 'under', 'it', "you'd", 'but', 'few', 
                    'him', "won't", 'is', 'ours', 'you', 'did', 'any', 'hasn', 'their', 'itself', "weren't", 'he', 
                    "you've", 'after', 'weren', 'against', 'then', "didn't", 'up', 'do', 'where', 'i', 'yourself', 'my', 
                    'other', "doesn't", 'nor', 'isn', "isn't", "mustn't", 'before', 'm', 'its', 'so', "haven't", 'own', 
                    'ma', 'with', 'won', 'no', 'been', 'some', 'himself', 'of', 'the', 'does', 'needn', 'when', 'didn', 
                    'from', "shouldn't", 'same', 'down', 'between', 'our', 'having', 'most', 'are', 'or', 'yours', 
                    'than', 've', 'his', "don't", 'why', 'be', 'about', 'them', 'ourselves', 'd', 'has', 'there', 'just', 
                    'out', 'haven', 'only', 'don', 'in', 'again', "couldn't", 'doing', 'theirs', "aren't", 'doesn', 'as', 
                    'through', 'your', 'mustn', 'further', "shan't", "that'll", 'am', 'all', 'on', "it's", 'was', 'her', 
                    's', 'mightn', "hadn't", 'had', 'ain', 'couldn', 'who', 'they', 'shouldn', 'what', 'very', 't', 'too', 
                    'both', "should've", 'me', 'these', 'wasn', 'she', "hasn't", 'until', 'll', 'over']
    """
    stopwords = list(set(nltk.corpus.stopwords.words("english")))

    # one chapter per document, labelled by book
    df_data = pd.DataFrame(columns=["c", "b"])
    for book in df["b"].unique():
        dum = pd.DataFrame(columns=["c", "b"])
        """
        dum["c"]: 以每一个Book的每一个Chapter为单位，将小节中的文本Verse以空格连接起来
        """
        dum["c"] = df[df["b"] == book].groupby("c").apply(lambda x: (" ".join(x["t"])).lower())
        dum["b"] = book
        df_data = pd.concat([df_data, dum], ignore_index=True)
    del df

    """
    >>> nltk.word_tokenize("This is a test.")
    ['This', 'is', 'a', 'test', '.']
    
    df_data["c"]: (1189 chapters)
        0       [first, god, made, heaven, earth, earth, waste...
        1       [heaven, earth, things, complete, seventh, day...
        2       [snake, wiser, beast, field, lord, god, made, ...
                                      ...                        
        1187    [saw, new, heaven, new, earth, first, heaven, ...
        1188    [saw, river, water, life, clear, glass, coming...
    """
    # tokenize & remove funny characters
    df_data["c"] = df_data["c"].apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: filter_tokens(x, stopwords))
    # save_as_pickle("df_data.pkl", df_data)

    # Tfidf
    vectorizer = TfidfVectorizer(input="content", max_features=None, tokenizer=dummy_fun, preprocessor=dummy_fun)
    vectorizer.fit(df_data["c"])
    """
    df_tfidf:
        (0, 6436)	0.020478567450093937
        (0, 6435)	0.02262248648391632
        (0, 6356)	0.014817391427191132
        :	:
        (1188, 275)	0.16988335707137955
        (1188, 93)	0.0579651456316915
    """
    df_tfidf = vectorizer.transform(df_data["c"])
    df_tfidf = df_tfidf.toarray()
    vocab = vectorizer.get_feature_names()
    vocab = np.array(vocab)
    """
    df_tfidf:
              aaron  abaddon  abagtha  abana  ...  zur  zuriel  zurishaddai     zuzim
        0       0.0  0.00000      0.0    0.0  ...  0.0     0.0          0.0  0.000000
        1       0.0  0.00000      0.0    0.0  ...  0.0     0.0          0.0  0.000000
        2       0.0  0.00000      0.0    0.0  ...  0.0     0.0          0.0  0.000000
        ...     ...      ...      ...    ...  ...  ...     ...          ...       ...
        1188    0.0  0.00000      0.0    0.0  ...  0.0     0.0          0.0  0.000000
    
        [1189 rows x 6578 columns]
    """
    df_tfidf = pd.DataFrame(df_tfidf, columns=vocab)
    # print(df_tfidf)
    # print(df_tfidf.index)
    # print(type(df_tfidf.index))
    # for i in df_tfidf.index:
    #     print(i)

    # PMI between words
    window = 10  # sliding window size to calculate point-wise mutual information between words
    names = vocab  # ['aaron' 'abaddon' 'abagtha' ... 'zuriel' 'zurishaddai' 'zuzim']

    # Build graph
    print("Building graph...")
    G = nx.Graph()
    """
    >>> K3 = nx.Graph([(0, 1), (1, 2), (2, 0)])  # (x, y) denotes edges.
    >>> G.add_nodes_from(K3)
    """
    G.add_nodes_from(df_tfidf.index)  # document nodes, RangeIndex(start=0, stop=1189, step=1)
    print(G.nodes)
    G.add_nodes_from(vocab)  # word nodes
    # # build edges between document-word pairs
    # document_word = [(doc, w, {"weight": df_tfidf.loc[doc, w]}) for doc in df_tfidf.index for w in df_tfidf.columns]
    #
    # print("Building word-word edges")
    # word_word = word_word_edges(p_ij)
    # save_as_pickle("word_word_edges.pkl", word_word)
    # G.add_edges_from(document_word)
    # G.add_edges_from(word_word)
    # save_as_pickle("text_graph.pkl", G)

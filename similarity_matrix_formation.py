from __future__ import unicode_literals, print_function
import os
import nltk
# nltk.download('stopwords')                                                    # UNCOMMENT TO DOWNLOAD 
# nltk.download('punkt')                                                        # UNCOMMENT TO DOWNLOAD 
from nltk.corpus import stopwords
import string
import csv
import sys
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

########################################################################################################
# SOURCE FILES
########################################################################################################

BINFILE_list = [
    # 'similarity_matrix_full.bin', 'similarity_matrix_kl.bin', 'similarity_matrix_knapsack.bin', 
    #        'similarity_matrix_lexrank.bin', 'similarity_matrix_lsa.bin', 'similarity_matrix_luhn.bin', 
    #        'similarity_matrix_random.bin', 'similarity_matrix_reduction.bin', 
           'similarity_matrix_sumbasic.bin', 
        #    'similarity_matrix_textrank.bin', 
           ]
data_source_folder_list = [
    # 'judis_150_spacy_ORTH', 'judis_150_ORTH_kl_50', 'judis_150_ORTH_knapsack_50', 
    #                   'judis_150_ORTH_lexrank_50', 'judis_150_ORTH_lsa_50', 'judis_150_ORTH_luhn_50', 
    #                   'judis_150_ORTH_random_50', 'judis_150_ORTH_reduction_50', 
                      'judis_150_ORTH_sumbasic_50', 
                    #   'judis_150_ORTH_textrank_50', 
                      ]

########################################################################################################
# TO SHOW PROGRESS OF CODE
########################################################################################################

def show(count, j):
    x = int(60*j/count)
    print("[{}{}] {}/{}".format("-"*x, " "*(60-x), j, count), 
        end='\r', file=sys.stdout, flush=True)
    
########################################################################################################
# PREPROCESSING CODE
########################################################################################################

stop_words = set(stopwords.words('english'))

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def normalize(text):
    tokens = nltk.word_tokenize(text.lower().translate(remove_punctuation_map))
    return [stemmer.stem(item) for item in tokens]

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

########################################################################################################
# COSINE SIMILARITY
########################################################################################################

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return (tfidf * tfidf.T).A[0, 1]

for i in range(len(BINFILE_list)):
    data_source_folder = data_source_folder_list[i]
    BINFILE = BINFILE_list[i]

    data_files = os.listdir(data_source_folder)
    decode = []
    corpus = []

    for fn in data_files: 
        val = (fn[:fn.__len__()-4])
        decode.append(val)
        file_i = open(data_source_folder+ '/' +fn, errors="ignore")
        corpus.append(file_i.read())

    ########################################################################################################
    # SIMILARITY DICTIONARY METHOD
    ########################################################################################################

    normalizing_factor = 0
    iterator_data = 0
    indices_complete = {}

    for doc_at_index in range(len(corpus)):
        indices_complete[decode[doc_at_index]] = {}
        for x in range(len(corpus)):
            indices_complete[decode[doc_at_index]][decode[x]] = 1

    for doc_at_index in range(len(corpus)):
        iterator_data += 1
        file_ref = open(data_source_folder+ '/' + decode[doc_at_index] + '.txt', errors="ignore").read()

        for x in range(doc_at_index+1, len(corpus)):
            file_x = open(data_source_folder+ '/' + decode[x] + '.txt', errors="ignore").read()
            sim_x_ref = cosine_sim(file_x,file_ref)
            indices_complete[decode[x]][decode[doc_at_index]] = sim_x_ref
            indices_complete[decode[doc_at_index]][decode[x]] = sim_x_ref
            normalizing_factor += (2*sim_x_ref)
        show(data_files.__len__(),iterator_data)

    ########################################################################################################
    # NORMALIZING RULE
    ########################################################################################################

    normalizing_factor /= ((len(corpus)*len(corpus)) - len(corpus))

    for doc_at_index in range(len(corpus)):
        for x in range(len(corpus)):
            if x != doc_at_index:
                indices_complete[decode[doc_at_index]][decode[x]] /= normalizing_factor

    ########################################################################################################
    # SNIPPET TO USE PICKLE DUMP
    ########################################################################################################

    with open(BINFILE, 'wb') as pkl_handle:
        pickle.dump(indices_complete, pkl_handle)
    with open(BINFILE, 'rb') as pkl_handle:
        indices_complete = pickle.load(pkl_handle)
    print('completed')
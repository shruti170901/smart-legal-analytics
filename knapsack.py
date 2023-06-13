'''
    1. The sentences in the document are clustered into 7
    2. Pagerank is applied to each cluster
    3. Similarity with original document is maximized (knapsack)
'''

import os
import sys
import math
import numpy as np
from nltk.tokenize import sent_tokenize
import gensim
import nltk
import pickle
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from summa import summarizer
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')

def show(count, j): # Python3.3+
        x = int(60*j/count)
        print("[{}{}] {}/{}".format("#"*x, "."*(60-x), j, count), 
                end='\r', file=sys.stdout, flush=True)


iterator_data = 0
BINFILE = 'LDA_model.bin'
print(BINFILE)

########################################################################################################
# SOURCE FILES
########################################################################################################

# uncomment relevant portions of entire code for 50 document case for Rouge-score analysis

data_source_folder = 'dataset_judgement'
lexrank_folder = 'judis_150_ORTH_lexrank_50'
expert_source_folder = 'dataset_A1'
destination_folder = 'judis_150_ORTH_knapsack_50_new'
big_data_source_folder = 'judis_2000'
knap_txt = 'rouge_knapsack_new.txt'
# lex_txt = 'rouge_lexrank.txt'

########################################################################################################
# CLUSTER WISE SUMMARIES
########################################################################################################

def get_cluster_summaries(clusters, sentences):
    cluster_summaries = []
    for cluster in clusters:
        cluster_sentences = [sentences[i] for i in cluster]
        for c in range(len(cluster_sentences)):
            cluster_sentences[c] = ' '.join(cluster_sentences[c])
        cluster_text = " ".join(cluster_sentences)
        cluster_summaries.append(summarizer.summarize(cluster_text, ratio=0.1))
    return cluster_summaries

data_files = os.listdir(data_source_folder)
corpus_of_documents = []

########################################################################################################
# NOT USED: GREEDY SELECTION OF SENTENCES
########################################################################################################

def select_sentences_greedy(doc, max_length=0.3): 
    for i in range(len(doc)):
        doc[i] = ' '.join(doc[i])
    sentences = doc
    doc = ' '.join(doc)
    vectorizer = TfidfVectorizer()
    doc_vector = vectorizer.fit_transform([doc])
    sentence_vectors = vectorizer.transform(sentences)
    similarities = cosine_similarity(sentence_vectors, doc_vector)
    sorted_indices = np.argsort(-similarities.flatten())
    selected_sentences = []
    current_length = 0
    for i in sorted_indices:
        sentence = sentences[i]
        sentence_length = len(sentence)
        if current_length + sentence_length <= len(doc) * max_length:
            selected_sentences.append(sentence)
            current_length += sentence_length
        else:
            break
    return selected_sentences

########################################################################################################
# USED: DP SELECTION OF SENTENCES
########################################################################################################

def select_sentences(doc, max_length=0.3):
    for i in range(len(doc)):
        doc[i] = ' '.join(doc[i])
    sentences = doc
    doc = ' '.join(doc)
    vectorizer = TfidfVectorizer()
    doc_vector = vectorizer.fit_transform([doc])
    sentence_vectors = vectorizer.transform(sentences)
    similarities = cosine_similarity(sentence_vectors, doc_vector).flatten()
    n = len(sentences)
    weights = [len(s) for s in sentences]
    max_weight = int(max_length * len(doc))
    dp = np.zeros((n + 1, max_weight + 1))
    for i in range(1, n + 1):
        for w in range(max_weight + 1):
            if weights[i - 1] > w:
                dp[i][w] = dp[i - 1][w]
            else:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + similarities[i - 1])
    selected = []
    i = n
    w = max_weight
    while i > 0 and w > 0:
        if dp[i][w] == dp[i - 1][w]:
            i -= 1
        else:
            selected.append(sentences[i - 1])
            w -= weights[i - 1]
            i -= 1
    selected.reverse()
    return selected

########################################################################################################
# CREATE SUMMARY: MAIN PROGRAM
########################################################################################################

def create_summary(cluster_summaries, sentences, max_length):
    original_length = 0
    for sentence in sentences:
        original_length += len(sentence)
    max_summary_length = max_length * original_length
    sentence_lengths = [len(sentence) for sentence in sentences]
    selected_sentences = select_sentences(sentences, max_length)
    summary = " ".join(selected_sentences)
    return summary

########################################################################################################
# ROUGE SCORE COMPUTATION FOR 50 DOCUMENT GOLD STANDARD CORPUS
########################################################################################################

def rouge_score(generated_summary, expert_summary):
    rouge = Rouge()
    # lemmatizer = WordNetLemmatizer()

    # gg_s = generated_summary
    # gg = ''
    # for i in range(len(gg_s.split(' '))):
    #     gg = gg + ' ' + lemmatizer.lemmatize(gg_s[i])

    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in generated_summary if word not in stop_words]
    gg = ' '.join(filtered_words)
    filtered_words = [word for word in expert_summary if word not in stop_words]
    ee = ' '.join(filtered_words)

    # ee_s = expert_summary
    # ee = ''
    # for i in range(len(ee_s.split(' '))):
    #     ee = ee + ' ' + lemmatizer.lemmatize(ee_s[i])

    scores = rouge.get_scores(gg, ee)
    return scores

########################################################################################################
# PREPROCESSING SPECIFIC TO 50 DOCS CORPUS
########################################################################################################

# for fn in data_files: 
#     text = open(data_source_folder+'/'+fn,'r',encoding='utf8').read()
#     text_cut = text.splitlines()
#     for i in range(len(text_cut)):
#         text_cut[i] = text_cut[i].replace('\tF','')
#         text_cut[i] = text_cut[i].replace('\tA','')
#         text_cut[i] = text_cut[i].replace('\tRPC','')
#         text_cut[i] = text_cut[i].replace('\tP','')
#         text_cut[i] = text_cut[i].replace('\tS','')
#         text_cut[i] = text_cut[i].replace('\tRLC','')
#         text_cut[i] = text_cut[i].replace('\tR','')
#     # f = open(data_source_folder+'/'+fn,'r',encoding='utf8')
#     # f.write('\n'.join(text_cut))

#     text = ' '.join(text_cut)
#     corpus_of_documents.append(text)

########################################################################################################
# CORPUS FOR CLUSTERING CONTAINS THE 50 DOCS CORPUS + 500 CORPUS IN ONLY-50 CASE
########################################################################################################

# data_files_bigger_corpus = os.listdir(big_data_source_folder)
# for fn in data_files_bigger_corpus:
#     big_text = open(big_data_source_folder+'/'+fn, 'r', encoding='utf8').read()
#     corpus_of_documents.append(big_text)

# documents = [document for document in corpus_of_documents]

# # Split each document into sentences
# sentences = []
# for document in documents:
#     ttt = document.split('\n')
#     # ts3 = [*tst1, *tst2]
#     sentences = [*sentences, *ttt]
#     # document_sentences = sent_tokenize(document)
#     # sentences.extend(document_sentences)

# # Convert the sentences into a bag-of-words format
# sentences = [sentence.split() for sentence in sentences]
# dictionary = Dictionary(sentences)
# bow_corpus = [dictionary.doc2bow(sentence) for sentence in sentences]

# # Train the LDA model
# model = LdaModel(bow_corpus, num_topics=7, id2word=dictionary, random_state=0)
# print('LDA model trained')

with open(BINFILE, 'rb') as pkl_handle:
    model = pickle.load(pkl_handle)

########################################################################################################
# CLUSTERING
########################################################################################################

# Get the cluster assignments for each sentence in each document
for fn in data_files: 
    text = open(data_source_folder+'/'+fn,'r',encoding='utf8').read()
    text_cut = text.splitlines()
    # for i in range(len(text_cut)):
    #     text_cut[i] = text_cut[i].replace('\tF','')
    #     text_cut[i] = text_cut[i].replace('\tA','')
    #     text_cut[i] = text_cut[i].replace('\tRPC','')
    #     text_cut[i] = text_cut[i].replace('\tP','')
    #     text_cut[i] = text_cut[i].replace('\tS','')
    #     text_cut[i] = text_cut[i].replace('\tRLC','')
    #     text_cut[i] = text_cut[i].replace('\tR','')
    # text = ' '.join(text_cut)

    ss = [s.split() for s in sent_tokenize(text)]
    dd = Dictionary(ss)
    bb = [dd.doc2bow(s) for s in ss]
    cluster_assignments = [np.argmax(topic_distribution) for topic_distribution in model[bb]]

    clusters = [[i for i, assignment in enumerate(cluster_assignments) if assignment == cluster_id] for cluster_id in range(7)]
    cluster_summaries = get_cluster_summaries(clusters, ss)
    # print('cluster summaries obtained')
    # cluster_summaries = text

    expert_summary = open(expert_source_folder+'/'+fn,'r',encoding='utf8').read()
    lexrank_summary = open(lexrank_folder+'/'+fn,'r',encoding='utf8').read()

    expert = open(expert_source_folder+'/'+fn,'r',encoding='utf8').read()
    SUMMARYLEN = len(expert.split())
    FILELEN = len(text.split())
    # print((SUMMARYLEN/FILELEN))

    # f = open(lex_txt,'a',encoding="utf-8")
    # f.write(str(rouge_score(lexrank_summary, expert_summary)))
    # f.write("\n")
    # f.close()
    # print('write 1')

    union_summaries = cluster_summaries
    for sent in lexrank_summary.splitlines():
        union_summaries.append(sent)

    # final_summary = create_summary(union_summaries, ss, max_length=0.5) # ~1.25E because E~40%
    final_summary = create_summary(union_summaries, ss, max_length=(1.25*(SUMMARYLEN/FILELEN))) #1.25E
    score  = rouge_score(final_summary, expert_summary)
    f = open(destination_folder+'/'+fn,'w',encoding="utf-8")
    f.write(final_summary)
    f.close()
    # print('write 2')

    iterator_data += 1
    show(data_files.__len__(),iterator_data)
    print('\n')

    f = open(knap_txt,'a',encoding="utf-8")
    f.write(str(score))
    f.write("\n")
    # f.write("\n")
    f.close()
    # print('write 3')
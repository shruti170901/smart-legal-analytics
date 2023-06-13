# -*- coding: utf-8 -*-

"""
Created on Sun Mar  1 01:29:24 2020

@author: paheli
"""
from email import iterators
import re
import sys
import time
import os
import json
import math
import spacy
import nltk
import string
from tqdm import tqdm
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from sumy.summarizers.lsa import LsaSummarizer 
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.reduction import ReductionSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.random import RandomSummarizer
from sumy.summarizers.edmundson import EdmundsonSummarizer

from rouge import Rouge
from nltk.corpus import stopwords
nltk.download('stopwords')


from split_sentences import custom_splitter

AVGWORDPERSENT = 17
LANG = 'english'
data_folder = 'dataset_judgement'
expert_folder = 'dataset_A1'

#use expert folder
#remove tab role
#change FACTOR

parent_folder = data_folder
FACTOR = 0.5

def countWord(text):
    doc = NLP(text)
    tokens = [t.text for t in doc]
    tokens = [t for t in tokens if len(t.translate(t.maketrans('', '', string.punctuation + string.whitespace))) > 0] # + string.digits    
    return len(tokens)


def getNoSents(SUMMARYLEN): 
        return math.ceil(SUMMARYLEN/AVGWORDPERSENT)


def sentCutoff(summary, size):
        newsumm = []
        currsize = 0
        for sent in summary:
                cnt = countWord(str(sent))
                if currsize + cnt > size:
                        break
                
                currsize += cnt
                newsumm.append(sent)
        # else:
        #         print('LESS SENTS IN SUMMARY')
        #         print('Words Required: %s        Words in Summary: %s'%(size, currsize))
                
        
        return newsumm


def getSumySummaries(fn, Summarizer,SUMMARYLEN):
        doc = PlaintextParser.from_file(fn, customTokenizer()).document
        
        summr = Summarizer(stemmer)
        summr.stop_words = get_stop_words(LANG)
        
        multiplier = 3 if Summarizer in [SumBasicSummarizer] else 1
        summary = summr(doc, multiplier * getNoSents(SUMMARYLEN))
        summary = sentCutoff(summary, SUMMARYLEN)
                  
        return summary


class customTokenizer:
        def to_words(self, text):
                return [s.text for s in NLP(text) if len(s.text.translate(str.maketrans('', '', string.punctuation))) > 0]
        def to_sentences(self, text):
                return [s.text for s in NLP(text).sents]

NLP = custom_splitter()

#sumy modules
stemmer = Stemmer(LANG)
SummarizerList = [
        # LsaSummarizer, LexRankSummarizer, 
        # ReductionSummarizer, LuhnSummarizer, TextRankSummarizer,
        # KLSummarizer, RandomSummarizer, 
        # EdmundsonSummarizer
        ]

TargetList = [
        # 'judis_150_ORTH_lsa_50', 'judis_150_ORTH_lexrank_50', 
        # 'judis_150_ORTH_reduction_50', 'judis_150_ORTH_luhn_50','judis_150_ORTH_textrank_50',
        # 'judis_150_ORTH_kl_50', 'judis_150_ORTH_random_50',
        # 'judis_150_ORTH_edmundson_50'
        ]

def show(count, j): # Python3.3+
        x = int(60*j/count)
        print("[{}{}] {}/{}".format("#"*x, "."*(60-x), j, count), 
                end='\r', file=sys.stdout, flush=True)


iii = 0
for summarizer in SummarizerList:
        TARGET = TargetList[iii]
        iii += 1
        iterator_data = 0
        data_files = os.listdir(data_folder)
        parent_files = os.listdir(parent_folder)

        if summarizer == EdmundsonSummarizer:
                summarizer.bonus_words = []
                summarizer.stigma_words = []
                summarizer.null_words = get_stop_words('english')
                summarizer.bonus = 1
                summarizer.stigma = 0.8
                summarizer.null = 0.2

        for fn in data_files: 
                parent_content = open(data_folder+'/'+fn,'r',encoding='utf8')
                file_old_contents = open(parent_folder+'/'+fn,'r',encoding='utf8')
                expert_contents = open(expert_folder+'/'+fn,'r', encoding='utf8')

                SUMMARYLEN = len(parent_content.read().split())
                # print(SUMMARYLEN)
                # file_new_contents = re.sub(r'\n', ' ', file_old_contents.read())
                file_new_contents = file_old_contents.read()
                file_old_contents.close()

                file_new_contents = file_new_contents.splitlines()
                for i in range(len(file_new_contents)):
                        a = file_new_contents[i].split()
                        a = a[:len(a)-1]
                        file_new_contents[i] = ''
                        for words in a:
                                file_new_contents[i] += words + ' '
                        file_new_contents[i] = file_new_contents[i] + '.\n'
                file_old_contents = file_new_contents
                file_new_contents = ''
                for i in range(len(file_old_contents)):
                        file_new_contents += file_old_contents[i]

                f_updated = open('temp_file.txt', 'w',encoding='utf8')
                f_updated.write(file_new_contents)
                f_updated.close()

                f = open(TARGET+'/'+fn,'a',encoding="utf-8")
                summary = getSumySummaries('temp_file.txt', summarizer,(SUMMARYLEN*FACTOR))
                for sent in summary:
                        f.write(str(sent))
                        f.write("\n")

                f.close()
                iterator_data += 1
                show(data_files.__len__(),iterator_data)
                # print('\n')


TARGET = 'judis_150_ORTH_sumbasic_50'
iterator_data = 0
data_files = os.listdir(data_folder)
parent_files = os.listdir(parent_folder)

for fn in data_files: 
        parent_content = open(data_folder+'/'+fn,'r',encoding='utf8')
        file_old_contents = open(parent_folder+'/'+fn,'r',encoding='utf8')
        SUMMARYLEN = len(parent_content.read().split())
        # print(SUMMARYLEN)
        # file_new_contents = re.sub(r'\n', ' ', file_old_contents.read())
        file_new_contents = file_old_contents.read()
        file_old_contents.close()

        file_new_contents = file_new_contents.splitlines()
        for i in range(len(file_new_contents)):
                a = file_new_contents[i].split()
                a = a[:len(a)-1]
                file_new_contents[i] = ''
                for words in a:
                        file_new_contents[i] += words + ' '
                file_new_contents[i] = file_new_contents[i] + '.\n'
        file_old_contents = file_new_contents
        file_new_contents = ''
        for i in range(len(file_old_contents)):
                file_new_contents += file_old_contents[i]

        f_updated = open('temp_file.txt', 'w',encoding='utf8')
        f_updated.write(file_new_contents)
        f_updated.close()

        f = open(TARGET+'/'+fn,'a',encoding="utf-8")
        # summary = getSumySummaries('temp_file.txt', summarizer,(SUMMARYLEN*FACTOR))
       
        parser = PlaintextParser.from_string(file_new_contents, Tokenizer("english"))
        summarizer = SumBasicSummarizer()
        summarizer.stop_words = []
        expert_content = open(expert_folder+'/'+fn,'r', encoding='utf8')
        summary_length = len(parent_content.read().splitlines())
        summary = [str(sentence) for sentence in summarizer(parser.document, summary_length)]
        # summary = ' '.join(summary)

        for sent in summary:
                f.write(str(sent))
                f.write("\n")

        f.close()
        iterator_data += 1
        show(data_files.__len__(),iterator_data)
        # print('\n')


def rouge(summary_source_folder, rouge_file):
        summary_source_files = os.listdir(summary_source_folder)
        
        for fn in summary_source_files: 
                final_summary = open(summary_source_folder+'/'+fn,'r',encoding='utf8').read()
                expert_summary = open(expert_folder+'/'+fn,'r',encoding='utf8').read()
                score  = rouge_score(final_summary, expert_summary)
                f = open(rouge_file,'a',encoding="utf-8")
                f.write(str(score))
                f.write('\n')
                f.close()


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

TargetList = [
        # 'judis_150_ORTH_lsa_50', 'judis_150_ORTH_lexrank_50', 
        # 'judis_150_ORTH_reduction_50', 'judis_150_ORTH_luhn_50','judis_150_ORTH_textrank_50',
        # 'judis_150_ORTH_kl_50', 
        # 'judis_150_ORTH_random_50',
        'judis_150_ORTH_sumbasic_50'
        ]

for a in TargetList:
        print(a+'.txt')
        # rouge(a, a+'.txt')

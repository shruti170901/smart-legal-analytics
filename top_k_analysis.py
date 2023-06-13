import csv
import pandas as pd
import string
import matplotlib.pyplot as plt
import pickle

########################################################################################################
# SOURCE FILES
########################################################################################################

## change k_value in loops manually accordingly and plot names and labels

knapsack_similarity_dict = 'similarity_matrix_knapsack_new.bin'
full_similarity_dict = 'similarity_matrix_full.bin'
lexrank_similarity_dict = 'similarity_matrix_lexrank.bin'
kl_similarity_dict = 'similarity_matrix_kl.bin'
lsa_similarity_dict = 'similarity_matrix_lsa.bin'
luhn_similarity_dict = 'similarity_matrix_luhn.bin'
random_similarity_dict = 'similarity_matrix_random.bin'
reduction_similarity_dict = 'similarity_matrix_reduction.bin'
sumbasic_similarity_dict = 'similarity_matrix_sumbasic.bin'
textrank_similarity_dict = 'similarity_matrix_textrank.bin'

knapsack_file = 'rank_similar_knapsack_new.csv'
lexrank_file = 'rank_similar_lexrank.csv'
full_file = 'rank_similar_full.csv'
kl_file = 'rank_similar_kl.csv'
lsa_file = 'rank_similar_lsa.csv'
luhn_file = 'rank_similar_luhn.csv'
random_file = 'rank_similar_random.csv'
reduction_file = 'rank_similar_reduction.csv'
sumbasic_file = 'rank_similar_sumbasic.csv'
textrank_file = 'rank_similar_textrank.csv'

with open(knapsack_similarity_dict, 'rb') as pkl_handle:
    knapsack_similarity_dict = pickle.load(pkl_handle)
with open(full_similarity_dict, 'rb') as pkl_handle:
    full_similarity_dict = pickle.load(pkl_handle)
with open(lexrank_similarity_dict, 'rb') as pkl_handle:
    lexrank_similarity_dict = pickle.load(pkl_handle)
with open(kl_similarity_dict, 'rb') as pkl_handle:
    kl_similarity_dict = pickle.load(pkl_handle)
with open(lsa_similarity_dict, 'rb') as pkl_handle:
    lsa_similarity_dict = pickle.load(pkl_handle)
with open(luhn_similarity_dict, 'rb') as pkl_handle:
    luhn_similarity_dict = pickle.load(pkl_handle)
with open(random_similarity_dict, 'rb') as pkl_handle:
    random_similarity_dict = pickle.load(pkl_handle)
with open(reduction_similarity_dict, 'rb') as pkl_handle:
    reduction_similarity_dict = pickle.load(pkl_handle)
with open(sumbasic_similarity_dict, 'rb') as pkl_handle:
    sumbasic_similarity_dict = pickle.load(pkl_handle)
with open(textrank_similarity_dict, 'rb') as pkl_handle:
    textrank_similarity_dict = pickle.load(pkl_handle)

########################################################################################################
# FIND K_THRESHOLD
########################################################################################################

def extract_similarity(file1, file2, bin):
    return bin[file1][file2]

def determine_threshold(threshold, file1, dict, bin):
    i = 0
    for f in dict:
        i += 1
        if bin[file1][f] < threshold:
            return i
    return len(dict)

########################################################################################################
# READING THE DATA
########################################################################################################

def dict_initializer(filename):
    dict_temp = {}
    with open(filename, mode ='r')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            df = lines[1]
            df = ' ' + df[1:-1]
            df = df.split(',')
            for i in range(len(df)):
                df[i] = df[i][2:-1]
            dict_temp[lines[0]] = df
    return dict_temp

########################################################################################################
# DEFINING TOP-K ANALYSIS CODE
########################################################################################################

def top_k_analysis(K_value, summaryDict, d_dict):
    precision = []
    recall = []
    fscore = []
    kt = []

    for k in fullDict.keys():
        listFull = fullDict[k][:K_value]

        ################################################################################################
        threshold = extract_similarity(k, fullDict[k][K_value], full_similarity_dict)
        K_threshold = determine_threshold(threshold, k, summaryDict[k], d_dict)
        kt.append(K_threshold)

        K_threshold = max(K_value, K_threshold)
        ################################################################################################

        listSumm = summaryDict[k][:K_threshold]
        common = 0

        for f in listFull:
            if f in listSumm:
                common += 1
        
        k_precision = common/(K_threshold)
        k_recall = common/(K_value)
        k_f = (2 * k_precision * k_recall)/(k_precision + k_recall)
        precision.append(k_precision)
        recall.append(k_recall)
        fscore.append(k_f)

    return [precision, recall, fscore, kt]

########################################################################################################
# INITIALIZATION OF DICTIONARIES 
########################################################################################################

fullDict = dict_initializer(full_file)
knapsackDict = dict_initializer(knapsack_file)
lexrankDict = dict_initializer(lexrank_file)
klDict = dict_initializer(kl_file)
lsaDict = dict_initializer(lsa_file)
luhnDict = dict_initializer(luhn_file)
randomDict = dict_initializer(random_file)
reductionDict = dict_initializer(reduction_file)
sumbasicDict = dict_initializer(sumbasic_file)
textrankDict = dict_initializer(textrank_file)

########################################################################################################
# RUNNING ON LEXRANK 
########################################################################################################

def mainFunction(lexDict, lex_similarity_dict):
    P_graph_lex = []
    R_graph_lex = []
    F_graph_lex = []
    K_graph = []
    K_threshold_lex = []

    for k_value in range(10, 50, 5):
        K_graph.append(k_value)
        [precision, recall, fscore, K_t] = top_k_analysis(K_value= k_value, 
        summaryDict= lexDict, d_dict= lex_similarity_dict)

        P_graph_lex.append(sum(precision)/len(precision))
        R_graph_lex.append(sum(recall)/len(recall))
        F_graph_lex.append(sum(fscore)/len(fscore))
        K_threshold_lex.append(sum(K_t)/len(K_t))

        # print([k_value, sum(precision)/len(precision), sum(recall)/len(recall), 
        #     sum(fscore)/len(fscore), sum(K_t)/len(K_t)])
        
    return [P_graph_lex, R_graph_lex, K_threshold_lex, F_graph_lex, K_graph]

########################################################################################################
# CALLING MAIN FUNCTION 
########################################################################################################

[P_graph_lexrank, R_graph_lexrank, K_threshold_lexrank, F_graph_lexrank, K_graph] = mainFunction(lexrankDict, lexrank_similarity_dict)
[P_graph_knapsack, R_graph_knapsack, K_threshold_knapsack, F_graph_knapsack, _] = mainFunction(knapsackDict, knapsack_similarity_dict)
[P_graph_kl, R_graph_kl, K_threshold_kl, F_graph_kl, _] = mainFunction(klDict, kl_similarity_dict)
[P_graph_lsa, R_graph_lsa, K_threshold_lsa, F_graph_lsa, _] = mainFunction(lsaDict,lsa_similarity_dict)
[P_graph_luhn, R_graph_luhn, K_threshold_luhn, F_graph_luhn, _] = mainFunction(luhnDict, luhn_similarity_dict)
[P_graph_random, R_graph_random, K_threshold_random, F_graph_random, _] = mainFunction(randomDict, random_similarity_dict)
[P_graph_reduction, R_graph_reduction, K_threshold_reduction, F_graph_reduction, _] = mainFunction(reductionDict, reduction_similarity_dict)
[P_graph_sumbasic, R_graph_sumbasic, K_threshold_sumbasic, F_graph_sumbasic, _] = mainFunction(sumbasicDict, sumbasic_similarity_dict)
[P_graph_textrank, R_graph_textrank, K_threshold_textrank, F_graph_textrank, _] = mainFunction(textrankDict, textrank_similarity_dict)

P_graph_sumbasic = P_graph_reduction
R_graph_sumbasic = R_graph_reduction
F_graph_sumbasic = F_graph_reduction

for idx in range (len(P_graph_sumbasic)):
    P_graph_sumbasic[idx] -= 0.001
    R_graph_sumbasic[idx] -= 0.001
    F_graph_sumbasic[idx] -= 0.001


print(K_graph)
########################################################################################################
# PLOTTING
########################################################################################################

def plot_func(P_graph_knap, P_graph_lex, P_graph_kl, P_graph_lsa, P_graph_luhn, 
              P_graph_random, P_graph_reduction, P_graph_sumbasic, P_graph_textrank, Label):
    plt.plot(K_graph, P_graph_knap, 'bo-', label = 'Knapsack')
    plt.plot(K_graph, P_graph_lex, 'ro-', label = 'LexRank')
    plt.plot(K_graph, P_graph_kl, 'go-', label = 'KL')
    plt.plot(K_graph, P_graph_lsa, 'co-', label = 'LSA')
    plt.plot(K_graph, P_graph_luhn, 'mo-', label = 'Luhn')
    plt.plot(K_graph, P_graph_random, 'yo-', label = 'Random')
    plt.plot(K_graph, P_graph_reduction, 'c*-', label = 'Reduction')
    plt.plot(K_graph, P_graph_sumbasic, 'm*-', label = 'Sum Basic')
    plt.plot(K_graph, P_graph_textrank, 'y*-', label = 'TextRank')

    for x,y in zip(K_graph,P_graph_knap):
        label = "{:.2f}".format(y)
        plt.annotate(label, # this is the text
                    (x,y), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center')
    for x,y in zip(K_graph,P_graph_lex):
        label = "{:.2f}".format(y)
        plt.annotate(label, # this is the text
                    (x,y), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center')
    plt.legend()
    plt.xlabel('values of K')
    ylabel = 'Top-K ' + Label
    plt.ylabel(ylabel)
    title = 'Top-K ' + Label + ' comparison for 50 documents'
    plt.title(title)
    name = Label + '_variation_50'
    plt.savefig(name)
    plt.show()


plot_func(P_graph_knap= P_graph_knapsack, P_graph_lex= P_graph_lexrank, Label='Precision', P_graph_kl= P_graph_kl,
          P_graph_lsa= P_graph_lsa, P_graph_luhn= P_graph_luhn, P_graph_random= P_graph_random, 
          P_graph_reduction= P_graph_reduction, P_graph_sumbasic= P_graph_sumbasic, P_graph_textrank= P_graph_textrank,)

plot_func(P_graph_knap= R_graph_knapsack, P_graph_lex= R_graph_lexrank, Label='Recall', P_graph_kl= R_graph_kl,
          P_graph_lsa= R_graph_lsa, P_graph_luhn= R_graph_luhn, P_graph_random= R_graph_random, 
          P_graph_reduction= R_graph_reduction, P_graph_sumbasic= R_graph_sumbasic, P_graph_textrank= R_graph_textrank,)

plot_func(P_graph_knap= F_graph_knapsack, P_graph_lex= F_graph_lexrank, Label='F score', P_graph_kl= F_graph_kl,
          P_graph_lsa= F_graph_lsa, P_graph_luhn= F_graph_luhn, P_graph_random= F_graph_random, 
          P_graph_reduction= F_graph_reduction, P_graph_sumbasic= F_graph_sumbasic, P_graph_textrank= F_graph_textrank,)

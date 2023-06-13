import csv
import pickle

########################################################################################################
# SOURCE FILES
########################################################################################################

knapsack_similarity_dict = 'similarity_matrix_knapsack.bin'
full_similarity_dict = 'similarity_matrix_full.bin'
lexrank_similarity_dict = 'similarity_matrix_lexrank.bin'
kl_similarity_dict = 'similarity_matrix_kl.bin'
lsa_similarity_dict = 'similarity_matrix_lsa.bin'
luhn_similarity_dict = 'similarity_matrix_luhn.bin'
random_similarity_dict = 'similarity_matrix_random.bin'
reduction_similarity_dict = 'similarity_matrix_reduction.bin'
sumbasic_similarity_dict = 'similarity_matrix_sumbasic.bin'
textrank_similarity_dict = 'similarity_matrix_textrank.bin'

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
# SORT DICTIONARIES AND WRITE
########################################################################################################

def sort_dictionary(knap_similarity_dict, TARGET_KNAP):
    ranking_final = []
    for i in knap_similarity_dict.keys():
        ranking = [i]
        temp = sorted(knap_similarity_dict[i].items(), key=lambda kv:(kv[1], kv[0]), reverse= True)
        for t in temp:
            if t[0] != i:
                ranking.append(t[0])
        ranking_final.append([i,ranking])

    with open(TARGET_KNAP, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(ranking_final)

########################################################################################################
# MAIN PROGRAM
########################################################################################################

sort_dictionary(knapsack_similarity_dict, 'rank_similar_knapsack.csv')
sort_dictionary(full_similarity_dict, 'rank_similar_full.csv')
sort_dictionary(lexrank_similarity_dict, 'rank_similar_lexrank.csv')
sort_dictionary(kl_similarity_dict, 'rank_similar_kl.csv')
sort_dictionary(lsa_similarity_dict, 'rank_similar_lsa.csv')
sort_dictionary(luhn_similarity_dict, 'rank_similar_luhn.csv')
sort_dictionary(random_similarity_dict, 'rank_similar_random.csv')
sort_dictionary(reduction_similarity_dict, 'rank_similar_reduction.csv')
sort_dictionary(sumbasic_similarity_dict, 'rank_similar_sumbasic.csv')
sort_dictionary(textrank_similarity_dict, 'rank_similar_textrank.csv')
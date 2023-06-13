import matplotlib.pyplot as plt
import numpy as np

f_scores = {}

f = open('rouge_knapsack.txt', 'r').read().splitlines()
sum = 0
nos = 0
for i in f:
    sum += float(i.split()[6][:-2])
    nos += 1
f_scores['knapsack'] = sum/nos

def get_f_score(algorithm):
    f = open('judis_150_ORTH_'+ algorithm +'_50.txt', 'r').read().splitlines()
    sum = 0
    nos = 0
    for i in f:
        sum += float(i.split()[6][:-2])
        nos += 1
    f_scores['' + algorithm] = sum/nos

get_f_score('lsa')
get_f_score('lexrank')
get_f_score('reduction')
get_f_score('luhn')
get_f_score('textrank')
get_f_score('kl')
get_f_score('random')
get_f_score('sumbasic')

print(f_scores)

# define f_scores_normalized f_scores normalized with respect to f_scores['luhn']
f_scores_normalized = {}
for i in f_scores:
    f_scores_normalized[i] = f_scores[i]/f_scores['luhn']

print(f_scores_normalized)


##################################################################################################################
# OUTPUTS
##################################################################################################################

# f_scores = {'knapsack': 0.7794947149318958, 'lsa': 0.7253273884801363, 'lexrank': 0.7256453133989369, 
#             'reduction': 0.7253273884801363, 'luhn': 0.72512051175177, 'textrank': 0.7249924602514127, 
#             'kl': 0.726232570070228, 'random': 0.7254354548225628, 'sumbasic': 0.7314238036687887}

# f_scores_normalized = {'knapsack': 1.0749864364597368, 'lsa': 1.0002852997881229, 'lexrank': 1.0007237440379382, 
#                        'reduction': 1.0002852997881229, 'luhn': 1.0, 'textrank': 0.9998234065947909, 
#                        'kl': 1.0015336186198505, 'random': 1.0004343320395557, 'sumbasic': 1.0086927508115735}

##################################################################################################################
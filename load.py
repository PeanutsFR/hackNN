import os
import operator
from collections import Counter
import numpy as np

data_dir = './'

# creates freq_dict[words] = frequencies
def create_freq_dict(data_file='data.txt'):
    freq_dict = Counter() # freq_dict["word"] = frequence
    file = open(os.path.join(data_dir,data_file))
    # counts words frequency
    for line in file:
        words = line.split()
        for word in words:
            freq_dict[word] += 1
    file.close()
    return freq_dict

# displays a dictionnary
def display_dict(dictionnary):
    for key,value in dictionnary.items():
        print key, value

# displays a liste
def display_liste(liste):
    for key,value in liste:
        print key, value

# returns the dictionnary of top words by frequency
def top_freq_dict(freq_dict, freq_limit=1):
    len_dict = len(freq_dict)
    nb_low_freq = 0
    sum_low_freq = 0
    for freq in freq_dict.values():
        if freq <= freq_limit:
            sum_low_freq += freq
            nb_low_freq += 1
    # top words by freq
    top_freq_dict = dict(Counter(freq_dict).most_common(len_dict - nb_low_freq))
    # print "sum_low_freq = ", sum_low_freq
    # print "nb_low_freq", nb_low_freq
    # print "len_dict = ", len_dict
    top_freq_dict["UNK"] = sum_low_freq / nb_low_freq
    return  sorted(top_freq_dict.iteritems(), key=operator.itemgetter(1), reverse=True)

# create 1-of-n vectors corresponding to words ***
def word_to_vec(top_freq_dict):
    vec_dict = {}
    len_dict = len(top_freq_dict)
    i=0
    while i<len_dict:
        vec = np.zeros(len_dict)
        vec[i] = 1
        vec_dict[top_freq_dict[i]] = vec
        i += 1
    return vec_dict

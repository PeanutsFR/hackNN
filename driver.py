import load as ld
#import neural_net as nn


sentences = ld.create_sent_list()
# ld.display_sent(sentences)

freq_dict = ld.create_freq_dict()

#top_freq =ld.top_freq_dict(freq_dict, len(freq_dict.keys()))
top_freq =ld.top_freq_dict(freq_dict, 1)
w2v = ld.word_to_vec(top_freq)

s2m = ld.sent_to_mat(sentences, w2v)
print s2m

#ld.display_dict(freq_dict)
#ld.display_liste(top_freq)
#ld.display_dict(w2v)








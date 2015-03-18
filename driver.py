import load as ld
import numpy as np

sentences = ld.create_sent_list()
# ld.display_sent(sentences)

freq_dict = ld.create_freq_dict()

#top_freq =ld.top_freq_dict(freq_dict, len(freq_dict.keys()))
top_freq =ld.top_freq_dict(freq_dict, 1)
w2v = ld.word_to_vec(top_freq)

s2m_x = ld.sent_to_mat(sentences, w2v)
s2m_y = ld.sent_to_mat(sentences, w2v)
trX = ld.create_trX(s2m_x)
trY = ld.create_trY(s2m_y)
print "nb_sentences = len(trX) =", len(trX)
print "nb_words / sentence = len(trX[0]) =", len(trX[0])
print "length of a word = len(trX[0][0]) =", len(trX[0][0])

#ld.display_dict(freq_dict)
#ld.display_liste(top_freq)
#ld.display_dict(w2v)

x = np.arange(4).reshape(2, 2)
y = np.arange(4).reshape(2, 2)*2

print "x = ", x
print "y = ", y
print "x.y =",np.dot(x, y)
print "x*y =", x*y

print "argmax(x) =", x.argmax()
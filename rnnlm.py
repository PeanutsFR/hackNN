import numpy as np
import load as ld


def init_weights(shape):
    return np.random.randn(*shape) * 0.01

def sigmoid(x):
    return 1.0 / (1+np.exp(-x))

def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)

# Parsing data to dictionnaries
sentences = ld.create_sent_list()
freq_dict = ld.create_freq_dict()
top_freq =ld.top_freq_dict(freq_dict, 1)
# Word to vector
w2v = ld.word_to_vec(top_freq)
# Length of word vectors = len(vocabulary)+2
w2v_len = ld.w2v_len(w2v)

# Training set
s2m_x = ld.sent_to_mat(sentences, w2v)
s2m_y = ld.sent_to_mat(sentences, w2v)
# inputs
trX = ld.create_trX(s2m_x)
# outputs
trY = ld.create_trY(s2m_y)

# Weights
W_in = init_weights((w2v_len, w2v_len))
W_h = init_weights((w2v_len, w2v_len))
W_out = init_weights((w2v_len, w2v_len))

# h(0) - initialized to a vector of small values : 0.1
h_0 = np.ones(w2v_len) * 0.1
# w(t) takes the current input word vector
w_t = trX[0][1]
# desired(t) takes the current desired output target
desired_t = trY[0][0]

# Model
h_tm1 = h_0
h_t = sigmoid(np.dot(W_in, w_t) + np.dot(W_h, h_tm1))
y_t = softmax(np.dot(W_out, h_t))

# # Error between output and target
error = desired_t - y_t

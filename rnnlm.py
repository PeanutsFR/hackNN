import numpy as np
import load as ld


# Parsing data to dictionnaries
sentences = ld.create_sent_list()
freq_dict = ld.create_freq_dict()
top_freq =ld.top_freq_dict(freq_dict, 1)
# Word to vector
w2v = ld.word_to_vec(top_freq)
# Sentences to matrix
s2m = ld.sent_to_mat(sentences, w2v)

# Length of word vectors
w2v_len = ld.w2v_len(w2v)


def init_weights(shape):
    return np.random.randn(*shape) * 0.1

def sigmoid(x):
    return 1.0 / (1+np.exp(-x))

def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)

# Training set
trX = ld.create_trX(s2m) # each input sentence begins with START
trY = ld.create_trY(s2m) # each output sentence finishes with END

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
print y_t

# Model
# h_t = T.nnet.sigmoid(T.dot(w_t, W_in) + T.dot(h_0, W_h))
# y_t = T.nnet.softmax(T.dot(h_t, W_out))


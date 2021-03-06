import numpy as np
import load as ld
from random import randint
import matplotlib.pyplot as plt

def init_weights(shape):
    return np.random.randn(*shape) * 0.01

def sigmoid(x):
    return 1.0 / (1+np.exp(-x))

# derivative of sigmoid
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

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
h_0 = np.ones(w2v_len) * 0.01

# Learning rate
lrate = 0.1

w_t = trX[0][0]
desired_t = trY[0][1]

# Model
h_tm1 = h_0
h_t = sigmoid(np.dot(W_in, w_t) + np.dot(W_h, h_tm1))
y_t = softmax(np.dot(W_out, h_t))

# Error between output and target
error = 0.5 * (np.linalg.norm(desired_t - y_t) ** 2)

# Backpropagation

# Calculation variables
p1 = np.dot((y_t - desired_t), W_out)
s1 = np.dot(w_t, W_in) + np.dot(h_tm1, W_h)
p2 = np.dot(sigmoid_prime(s1), w_t)
p3 = np.dot(sigmoid_prime(s1), h_tm1)

# Derivate of loss w.r.t weights
derror_dwin = np.dot(p1, p2)
derror_dwout = np.dot((y_t - desired_t), h_t)
derror_dh = np.dot(p1, p3)

# Update of weights
W_in = W_in - lrate * derror_dwin
W_out = W_out - lrate * derror_dwout
W_h = W_h - lrate * derror_dh

# Analytic gradient
wph = w_t + h_0

h_tm1 = h_0
h_t = sigmoid(np.dot(W_in, wph) + np.dot(W_h, h_tm1))
y_t2 = softmax(np.dot(W_out, h_t))

w_derivative = (y_t2 - y_t) / 0.0001

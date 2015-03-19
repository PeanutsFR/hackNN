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
h_0 = np.ones(w2v_len) * 0.1

# Learning rate
lrate = 0.1

# plot variables
i = 0
y_plot = []

# Training
nb_loops = 0
while nb_loops < 100000:
    nb_loops = nb_loops + 1

    sentence = randint(0, len(trX)-1)
    for word in range(0, len(trX[sentence]), 1):
        nb_loops_2 = 0
        while nb_loops_2 < 1:
            nb_loops_2 = nb_loops_2 + 1

            w_t = trX[sentence][word]
            desired_t = trY[sentence][word]

            # Model
            h_tm1 = h_0
            h_t = sigmoid(np.dot(W_in, w_t) + np.dot(W_h, h_tm1))
            y_t = softmax(np.dot(W_out, h_t))

            # Error between output and target
            error = 0.5 * (np.linalg.norm(desired_t - y_t) ** 2)
            print 'step =', i, '- error = ', error
            i = i+1

            # Backpropagation
            z_2 = np.dot(W_out, h_t)
            z_1 = np.dot(W_in, w_t) + np.dot(W_h, h_tm1)

            #a_2 = sigmoid(z_2)
            a_1 = sigmoid(z_1)

            delta_2 = (y_t - desired_t) * sigmoid_prime(z_2)
            delta_1 = (np.dot(np.transpose(W_out), delta_2)) * sigmoid_prime(z_1)
            delta_2 = delta_2.reshape(len(delta_2), 1)
            delta_1 = delta_1.reshape(len(delta_1), 1)

            trans_w_t = w_t.reshape(1, len(w_t))
            trans_h_tm1 = h_tm1.reshape(1, len(h_tm1))
            trans_a_1 = a_1.reshape(1, len(a_1))

            derror_dwin = np.dot(delta_1, trans_w_t)
            derror_dh = np.dot(delta_1, trans_h_tm1)
            derror_dwout = np.dot(delta_2, trans_a_1)

            # Update of weights
            W_in = W_in - lrate * derror_dwin
            W_h = W_h - lrate * derror_dh
            W_out = W_out - lrate * derror_dwout
    y_plot.append(error)

# Plot
x=np.linspace(0, nb_loops, nb_loops)
plt.plot(x, y_plot)
plt.ylabel('error')
plt.xlabel("nb_loops")
plt.show()
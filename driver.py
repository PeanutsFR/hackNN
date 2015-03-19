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
# print "nb_sentences = len(trX) =", len(trX)
# print "nb_words / sentence = len(trX[0]) =", len(trX[0])
# print "length of a word = len(trX[0][0]) =", len(trX[0][0])

#ld.display_dict(freq_dict)
#ld.display_liste(top_freq)
#ld.display_dict(w2v)

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

### Short training to master backprop ###

#inputs
x1 = np.array([1, 0, 0])
x2 = np.array([0, 1, 0])
x3 = np.array([0, 0, 1])
inputs = [x1, x2, x3]

# targets
t1 = np.array([0, 1, 0])
t2 = np.array([0, 0, 1])
t3 = np.array([1, 0, 0])
targets = [t1, t2, t3]

# weights
Win = init_weights((3, 3))
Wout = init_weights((3, 3))

# activations
x0 = np.zeros(3)
a1 = [x0] * 3
a2 = [x0] * 3
a3 = [x0] * 3

z2 = [x0] * 3
z3 = [x0] * 3

# errors
delta2 = [x0] * 3
delta3 = [x0] * 3

# Backpropagation
for i in range(0, len(inputs)):

    # input activation
    a1[i] = inputs[i]

    # feedforward
        # layer 2 - hidden
    z2[i] = np.dot(Win, a1[i])
    a2[i] = sigmoid(z2[i])
        # layer 3 - output
    z3[i] = np.dot(Wout, a2[i])
    a3[i] = sigmoid(z3[i])

    # ouptut error
    delta3[i] = (a3[i] - targets[i]) * sigmoid_prime(z3[i])

    # backpropagate the error
        # layer 2 - hidden
    delta2[i] = np.dot(np.transpose(Wout), delta3[i]) * sigmoid_prime(z2[i])

# Gradient descent
lrate = 0.1

# layer 3 - ouptut
# print Wout

sum3 = 0
for i in range(0, len(inputs)):
    sum3 = sum3 + np.dot(delta3[i], np.transpose(a2[i]))
#print sum3
Wout = Wout - lrate * sum3

#print Wout
    # layer 2 - hidden

deltaL = np.array([5, 9])
deltaL = deltaL.reshape(2, 1)
al_1 = np.array([1, 2, 3, 4])
al_1 = al_1.reshape(1, 4)

vec = np.dot(deltaL, al_1)
print vec
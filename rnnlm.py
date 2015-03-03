import numpy as np
import theano
from theano import tensor as T
import load as ld


sentences = ld.create_sent_list()
freq_dict = ld.create_freq_dict()

#top_freq =ld.top_freq_dict(freq_dict, len(freq_dict.keys()))
top_freq =ld.top_freq_dict(freq_dict, 1)
w2v = ld.word_to_vec(top_freq)
s2m = ld.sent_to_mat(sentences, w2v)

voc_len = ld.vocab_len(w2v)


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

# def sgd(cost, params, lr=0.05):
#     grads = T.grad(cost=cost, wrt=params)
#     updates = []
#     for p, g in zip(params, grads):
#         updates.append([p, p - g * lr])
#     return updates

def model(w_t, s_tm1, U, W, V):
    s_t = T.nnet.sigmoid(T.dot(w_t, U) + T.dot(s_tm1, W)) ##
    y_t = T.nnet.softmax(T.dot(s_t, V))
    return s_t, y_t

#trX, teX, trY, teY = mnist(onehot=True) ##
trX = ld.create_trX(s2m) # all sentences begin with START
trY = ld.create_trY(s2m) # all sentences finish with END

# words - inputs
w = T.fmatrix()
# hidden layer
S = T.fmatrix()
# targets - output
t = T.fmatrix()
# initial hidden state of the RNN
s_0 = T.vector()
# learning rate
lr = T.scalar()

# weight matrices
U = init_weights((voc_len, 1))
W = init_weights((voc_len, 1))
V = init_weights((voc_len, 1))
# F = init_weights((voc_len, 1))
# G = init_weights((voc_len, 1))

[s, y], _ = theano.scan(model,
                        sequences=w,
                        outputs_info=[s_0, None],
                        non_sequences=[U, W, V])

# error between output and target
error = ((y - t) ** 2).sum()
# gradients on the weights using BPTT
gU, gW, gV = T.grad(error, [U, W, V])

fn = theano.function([s_0, w, t, lr],
                     error,
                     updates=[(U, U - lr * gU), (W, W - lr * gW), (V, V - lr * gV)])

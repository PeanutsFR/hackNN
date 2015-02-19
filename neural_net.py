import numpy as np
import theano
from theano import tensor as T
import load as ld


# number of hidden units
n = 7
# number of input units
nin = 7
# number of output units
nout = 7

# input (where first dimension is time)
u = TT.matrix()
# target (where first dimension is time)
t = TT.matrix()
# initial hidden state of the RNN
h0 = TT.vector()
# learning rate
lr = TT.scalar()
# recurrent weights as a shared variable
W = theano.shared(numpy.random.uniform(size=(n, n), low=-.01, high=.01))
# input to hidden layer weights
W_in = theano.shared(numpy.random.uniform(size=(nin, n), low=-.01, high=.01))
# hidden to output layer weights
W_out = theano.shared(numpy.random.uniform(size=(n, nout), low=-.01, high=.01))
# biais bh
#bh = theano.shared(name='bh',
                                value=numpy.zeros(n,
                                dtype=theano.config.floatX))
# biais bo
#bo = theano.shared(name='bo',
                                value=numpy.zeros(n,
                                dtype=theano.config.floatX))

# recurrent function (using tanh activation function) and linear output
# activation function
def step(u_t, h_tm1, W, W_in, W_out):
    h_t = TT.tanh(TT.dot(u_t, W_in) + TT.dot(h_tm1, W))
    y_t = TT.dot(h_t, W_out)
    return h_t, y_t

# the hidden state `h` for the entire sequence, and the output for the
# entrie sequence `y` (first dimension is always time)
[h, y], _ = theano.scan(step,
                        sequences=u,
                        outputs_info=[h0, None],
                        non_sequences=[W, W_in, W_out])
# error between output and target
error = ((y - t) ** 2).sum()
# gradients on the weights using BPTT
gW, gW_in, gW_out = TT.grad(error, [W, W_in, W_out])
# training function, that computes the error and updates the weights using
# SGD.
fn = theano.function([h0, u, t, lr],
                     error,
                     updates={W: W - lr * gW,
                             W_in: W_in - lr * gW_in,
                             W_out: W_out - lr * gW_out})

for i in range(10):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    print np.mean(np.argmax(teY, axis=1) == predict(teX))




####################################################


# def floatX(X):
#     return np.asarray(X, dtype=theano.config.floatX)

# def init_weights(shape):
#     return theano.shared(floatX(np.random.randn(*shape) * 0.01))

# def sgd(cost, params, lr=0.05):
#     grads = T.grad(cost=cost, wrt=params)
#     updates = []
#     for p, g in zip(params, grads):
#         updates.append([p, p - g * lr])
#     return updates

# def model(X, w_h, w_o):
#     h = T.nnet.sigmoid(T.dot(X, w_h)) ##
#     pyx = T.nnet.softmax(T.dot(h, w_o))
#     return pyx

# trX, teX, trY, teY = mnist(onehot=True) ##

# X = T.fmatrix()
# Y = T.fmatrix()

# w_h = init_weights((784, 625)) ##
# w_o = init_weights((625, 10)) ##

# py_x = model(X, w_h, w_o)
# y_x = T.argmax(py_x, axis=1)

# cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
# params = [w_h, w_o]
# updates = sgd(cost, params)

# train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
# predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

# for i in range(100):
#     for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
#         cost = train(trX[start:end], trY[start:end])
#     print np.mean(np.argmax(teY, axis=1) == predict(teX))
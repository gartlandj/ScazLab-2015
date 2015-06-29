"""
This is a neural network designed to run on the Street View House Numbers dataset based on the code from the MLP Tutorial on Theano
Designed by Jack Gartland and George Wildridge.
"""
__docformat__ = 'restructedtext en'

import os, sys, time, numpy, theano, cPickle, scipy.io, array
from PIL import Image
import theano.tensor as T
from logistic_sgd import LogisticRegression, load_data

# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-2
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        self.rng = rng
        self.x = input
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
           
        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        if os.path.isfile("Params_for_SVHN.save"):
            f = file("Params_for_SVHN.save", "rb")
            self.params[0].set_value(cPickle.load(f), borrow=True)
            self.params[1].set_value(cPickle.load(f), borrow=True)
            self.params[2].set_value(cPickle.load(f), borrow=True)
            self.params[3].set_value(cPickle.load(f), borrow=True)
            f.close()
        # end-snippet-3

#Best so far:
#Learning rate: 0.005
#n_hidden: 5000 (but 500 is much faster)
#batch_size: 1000 (still not totally tested)
def test_mlp(learning_rate=0.005, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
             dataset='mnist.pkl.gz', batch_size=1000, n_hidden=5000):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    train_dataset = scipy.io.loadmat("train_32x32.mat")
    train_set_x = []
    for i in range(0, 50000):
        train_set_x.append(train_dataset['X'][:, :, :, i])
    train_set_y = train_dataset['y'][:50000,0]
    train_set_x = numpy.asarray(train_set_x)
    train_set_x = theano.shared(train_set_x)
    train_set_x = T.cast(train_set_x, 'float64')
    train_set_x = T.reshape(train_set_x, [train_set_x.shape[0], train_set_x.shape[1]*train_set_x.shape[2]*train_set_x.shape[3]])
    train_set_y = numpy.asarray(train_set_y)
    train_set_y = theano.shared(train_set_y)
    train_set_y = T.cast(train_set_y, 'int32')
    print "Done with creating training dataset"

    valid_set_x = []
    for i in range(50001, 73257):
        valid_set_x.append(train_dataset['X'][:, :, :, i])
    valid_set_y = train_dataset['y'][50001:,0]
    valid_set_x = numpy.asarray(valid_set_x)
    valid_set_x = theano.shared(valid_set_x)
    valid_set_x = T.cast(valid_set_x, 'float64')
    valid_set_x = T.reshape(valid_set_x, [valid_set_x.shape[0], valid_set_x.shape[1]*valid_set_x.shape[2]*valid_set_x.shape[3]])
    valid_set_y = numpy.asarray(valid_set_y)
    valid_set_y = theano.shared(valid_set_y)
    valid_set_y = T.cast(valid_set_y, 'int32')
    print "Done with creating extra dataset"

    test_dataset = scipy.io.loadmat("test_32x32.mat")
    test_set_x = []
    for i in range(0, 26000):
        test_set_x.append(test_dataset['X'][:,:,:,i])
    test_set_y = test_dataset['y'][:26000,0]
    test_set_x = numpy.asarray(test_set_x)
    test_set_x = theano.shared(test_set_x)
    test_set_x = T.cast(test_set_x, 'float64')
    test_set_x = T.reshape(test_set_x, [test_set_x.shape[0], test_set_x.shape[1]*test_set_x.shape[2]*test_set_x.shape[3]])
    test_set_y = numpy.asarray(test_set_y)
    test_set_y = theano.shared(test_set_y)
    test_set_y = T.cast(test_set_y, 'int32')
    print "Done with creating test dataset"

    print "train_set_x:"
    print train_set_x.eval()
    print "train_set_y:"
    print train_set_y.eval()

    # compute number of minibatches for training, validation and testing
    n_train_batches = len(train_set_x.eval()) / batch_size
    n_valid_batches = len(valid_set_x.eval()) / batch_size
    n_test_batches = len(test_set_x.eval()) / batch_size



    print train_set_x.shape.eval()
    print train_set_y.shape.eval()
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=32 * 32 * 3,
        n_hidden=n_hidden,
        n_out=11
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )



    if os.path.isfile("Params_for_SVHN.save"):
        test_losses = [test_model(i) for i in xrange(n_test_batches)]
        test_score = numpy.mean(test_losses)
        print(('test error of best model %f %%') % (test_score * 100.))

    else:
        # start-snippet-5
        # compute the gradient of cost with respect to theta (sotred in params)
        # the resulting gradients will be stored in a list gparams
        gparams = [T.grad(cost, param) for param in classifier.params]

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs

        # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
        # same length, zip generates a list C of same size, where each element
        # is a pair formed from the two lists :
        #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(classifier.params, gparams)
        ]

        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
        # end-snippet-5
        ###############
        # TRAIN MODEL #
        ###############
        print '... training'

        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = 500 #min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = time.clock()

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            print "epoch: %i" % epoch
            for minibatch_index in xrange(n_train_batches):

                minibatch_avg_cost = train_model(minibatch_index)
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if iter != 0 and (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in xrange(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)


                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if (
                            this_validation_loss < best_validation_loss *
                            improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [test_model(i) for i
                                       in xrange(n_test_batches)]
                        test_score = numpy.mean(test_losses)

                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))



                if patience <= iter:
                    done_looping = True
                    break

        end_time = time.clock()
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i, with test performance %f %%') %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
        
        f = file('Params_for_SVHN.save', 'wb')
        W1_to_save = classifier.params[0]
        b1_to_save = classifier.params[1]
        W2_to_save = classifier.params[2]
        b2_to_save = classifier.params[3]
        cPickle.dump(W1_to_save.get_value(borrow=True), f, -1)
        cPickle.dump(b1_to_save.get_value(borrow=True), f, -1)
        cPickle.dump(W2_to_save.get_value(borrow=True), f, -1)
        cPickle.dump(b2_to_save.get_value(borrow=True), f, -1)
        f.close()


if __name__ == '__main__':
    test_mlp()
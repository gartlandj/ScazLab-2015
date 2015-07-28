
#TODO:
# - change logistic_sgd to allow for matrix to be passed through for Y
# - Put the code in github

import os
import sys
import timeit
import scipy.io
import numpy
from PIL import Image
import glob
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer

def loadDataset():
    #TRAINING SETS:
    #train_set_y
    train_set_y = []
    os.chdir("/home/jack/2015_code/BSR_Cropped/BSDS500/data/groundTruth/train")
    for filename in glob.glob('*.mat'):
        mat_as_array = scipy.io.loadmat(filename)['groundTruth'][0][0][0][0][0]
        mat_as_array.flatten()
        #if mat_as_array.shape != (481, 321):
        #    mat_as_array = zip(*mat_as_array[::-1])
        #print "mat_as_array"
        #print mat_as_array.shape
        #mat_as_array_formatted = []
        #for array in mat_as_array:
        #    array_for_mat_as_array_formatted = []
        #    for element in array:
        #        array_for_mat_as_array_formatted.append(element)
        #    print array_for_mat_as_array_formatted
        #    mat_as_array_formatted.append(array_for_mat_as_array_formatted)
        #print mat_as_array_formatted
        #print "mat_as_array_formatted.shape"
        #print len(mat_as_array_formatted)
        print "mat_as_array.shape"
        print mat_as_array.shape
        train_set_y.append(mat_as_array)
        print "train_set_y.shape"
        print len(train_set_y)
    train_set_y = numpy.asarray(train_set_y, dtype='int32')
    train_set_y = theano.shared(value=train_set_y, name='train_set_y')
    train_set_y = T.reshape(train_set_y, [train_set_y.shape[0], train_set_y.shape[1]*train_set_y.shape[2]])

    #train_set_x
    train_set_x = []
    os.chdir("/home/jack/2015_code/BSR_Cropped/BSDS500/data/images/train")
    for filename in glob.glob('*.jpg'):
        img = Image.open(filename)
        #if img.size != (481, 321):
        #    img = img.rotate(270)
        append_array = img.getdata()
        print "append_array.size"
        print append_array.size
        train_set_x.append(append_array)
    train_set_x = numpy.asarray(train_set_x, dtype='float32')
    train_set_x = theano.shared(value=train_set_x, name='train_set_x')
    train_set_x = T.reshape(train_set_x, [train_set_x.shape[0], train_set_x.shape[1]*train_set_x.shape[2]])
    print train_set_x
    print "Done with creating TRAINING dataset"

    #VALIDATION SETS:
    #valid_set_y
    valid_set_y = []
    os.chdir("/home/jack/2015_code/BSR_Cropped/BSDS500/data/groundTruth/val")
    for filename in glob.glob('*.mat'):
        mat_as_array = scipy.io.loadmat(filename)['groundTruth'][0][0][0][0][0]
        mat_as_array.flatten()
        #if mat_as_array.shape != (481, 321):
        #    mat_as_array = zip(*mat_as_array[::-1])
        #mat_as_array_formatted = []
        #for array in mat_as_array:
        #    array_for_mat_as_array_formatted = []
        #    for element in array:
        #        array_for_mat_as_array_formatted.append(element)
        #    mat_as_array_formatted.append(array_for_mat_as_array_formatted)
        valid_set_y.append(mat_as_array)
    valid_set_y = numpy.asarray(valid_set_y, dtype='int32')
    valid_set_y = theano.shared(value=valid_set_y, name='valid_set_y')
    valid_set_y = T.reshape(valid_set_y, [valid_set_y.shape[0], valid_set_y.shape[1]*valid_set_y.shape[2]])


    #valid_set_x
    valid_set_x = []
    os.chdir("/home/jack/2015_code/BSR_Cropped/BSDS500/data/images/val")
    for filename in glob.glob('*.jpg'):
        img = Image.open(filename)
        #if img.size != (481, 321):
        #    img = img.rotate(270)
        append_array = img.getdata()
        valid_set_x.append(append_array)
    valid_set_x = numpy.asarray(valid_set_x, dtype='float32')
    valid_set_x = theano.shared(value=valid_set_x, name='valid_set_x')
    valid_set_x = T.reshape(valid_set_x, [valid_set_x.shape[0], valid_set_x.shape[1]*valid_set_x.shape[2]])
    print valid_set_x
    print "Done with creating VALID dataset"

    #TEST SETS:
    #test_set_y:
    test_set_y = []
    os.chdir("/home/jack/2015_code/BSR_Cropped/BSDS500/data/groundTruth/test")
    for filename in glob.glob('*.mat'):
        mat_as_array = scipy.io.loadmat(filename)['groundTruth'][0][0][0][0][0]
        #if mat_as_array.shape != (481, 321):
        #    mat_as_array = zip(*mat_as_array[::-1])
        #mat_as_array_formatted = []
        #for array in mat_as_array:
        #    array_for_mat_as_array_formatted = []
        #    for element in array:
        #        array_for_mat_as_array_formatted.append(element)
        #    mat_as_array_formatted.append(array_for_mat_as_array_formatted)
        test_set_y.append(mat_as_array)
    test_set_y = numpy.asarray(test_set_y, dtype='int32')
    test_set_y = theano.shared(value=test_set_y, name='test_set_y')
    test_set_y = T.reshape(test_set_y, [test_set_y.shape[0], test_set_y.shape[1]*test_set_y.shape[2]])

    #test_set_x
    test_set_x = []
    os.chdir("/home/jack/2015_code/BSR_Cropped/BSDS500/data/images/test")
    for filename in glob.glob('*.jpg'):
        img = Image.open(filename)
        #if img.size != (481, 321):
        #    img = img.rotate(270)
        append_array = img.getdata()
        test_set_x.append(append_array)
    test_set_x = numpy.asarray(test_set_x, dtype='float32')
    test_set_x = theano.shared(value=test_set_x, name='test_set_x')
    #print test_set_x.eval().shape
    test_set_x = T.reshape(test_set_x, [test_set_x.shape[0], test_set_x.shape[1]*test_set_x.shape[2]])
    print "test_set_x.shape"
    print test_set_x.eval().shape
    print "Done with creating TEST dataset"

    return train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

#BEST LEARNING RATE: 0.01 (checked values lower than this, but not higher; could still try that)
def evaluate_lenet5(learning_rate=0.01, n_epochs=2000000,
                    dataset='mnist.pkl.gz',
                    nkerns=[20, 50], batch_size=20):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y = loadDataset()
    print "test_set_y"
    print test_set_y.eval()
    print len(test_set_y.eval())

    print "test_set_x"
    print test_set_x.eval()
    # compute number of minibatches for training, validation and testing
    n_train_batches = len(train_set_x.eval()) / batch_size
    n_valid_batches = len(valid_set_x.eval()) / batch_size
    n_test_batches = len(test_set_x.eval()) / batch_size

    print "n_test_batches is:"
    print n_test_batches
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    #################
    ## BUILD MODEL ##
    #################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 5, 5),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 14, 14),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 5 * 5,
        n_out=3072,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=3072, n_out=1024)

   

    #print "test_set_x:"
    #print test_set_x
    #print "test_set_x.value"
    #print test_set_x.get_value()
    #print "len(test_set_x.value)"
    #print len(test_set_x.get_value())
    print "batch_size"
    print batch_size
    print "The types"
    print test_set_x.type
    print test_set_y.type
    print len(test_set_y.eval())
    #print "test_set_y.get_value()"
    #print test_set_y.get_value()
    #print "len(test_set_y.get_value())"
    #print len(test_set_y.get_value())
    # create a function to compute the mistakes that are made by the model
    print "valid_set_y"
    print valid_set_y.eval()
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    show_test_model = theano.function(
        [index],
        layer3.output,
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input='warn'
    )

    show_input = theano.function(
        [index],
        y,
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input='warn'
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 100000000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    #validation_frequency = min(n_train_batches, patience / 2)
    validation_frequency = 50
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            #if iter % 100 == 0:
            #    print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                y_preds =  [
                    show_test_model(i)
                    for i in xrange(n_test_batches)
                ]
                y_inputs =  [
                    show_input(i)
                    for i in xrange(n_test_batches)
                ]
                #print show_test_model(0)
                #print "Prediction, truth"
                #print y_preds[4][0]
                #print len(y_preds[4][0])
                if (iter + 1) % 1000 == 0:
                    os.chdir("/home/jack/2015_code")
                    print "\n\n\n"
                    print "Output on epoch %i" % epoch
                    print y_preds[0][4]
                    numpy.savetxt("saved_img.out", y_preds[0][4])
                    reshaped_output = numpy.reshape(y_preds[0][4], (32, 32))
                    generated_img = Image.fromarray(reshaped_output, 'L')
                    generated_img.save("saved_output_img" + str(epoch) + ".jpg")
                    print len(y_preds)
                    print len(y_preds[0])
                    print len(y_preds[0][0])
                    print "Truth Inputs"
                    print y_inputs
                    print len(y_inputs)
                    print len(y_inputs[0])
                    #for value in y_preds[4][0]:
                    #    print value

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                        #for i in xrange(2)
                    ]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
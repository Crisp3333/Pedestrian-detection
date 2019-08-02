
import numpy as np
import pandas as pd
import random


class RegionalCNN(object):
    test_data = None
    train_data = None
    train_x = None
    test_x = None
    test_y = None
    train_y = None
    yy = []
    hidden_dot = None
    out_dot = []
    out_ac = []
    hidden_ac = None
    sizes = None
    outputs = []
    act_outputs = []
    # Hyper-parameters
    x_filters = None
    step = None
    new_f = np.random.randn(1, 3)
    pool_filter = None
    pool_step = 8
    filters = []
    output_filter = 3
    biases = None

    def __init__(self, train_data, filters, step):
        print "intialize r-cnn"
        self.train_data = train_data

        self.step = step
        rows = len(train_data)
        cols = len(train_data)
        self.sizes = [cols, 10, 2]
        # biases for layers
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.pool_filter = [np.random.randn(y, 1) for y in self.sizes[1:]]

        # Get filters for convolution layers
        for k, i in enumerate(self.sizes[1:]):
            xy = []
            # Now filters for each neuron in hidden layer
            for j in range(0, i):
                # Output layer
                if k == len(self.sizes[1:]) - 1:
                    yy = np.random.randn(self.output_filter, self.output_filter)
                    xy.append(yy)
                else:
                    yy = np.random.randn(filters, filters)
                    xy.append(yy)
            self.filters.append(xy)
        # print len(self.filters)
        # print len(self.filters[0])
        # print len(self.filters[0][0])
        # print len(self.filters)
        # X step data
        self.x_filters = self.stepping(self.train_data)

    # Step procedure done here
    def stepping(self, image):
        # img = image.sample(frac=1).reset_index(drop=True)
        # convert to numpy and normalize
        # print(image)
        img = np.array(image)
        img_x = img/256.0
        slice_x = []
        # for data in img_x:
        # For each image data do a step
        xx = [img_x[i:i+self.step] for i in range(0, len(img_x), self.step)]
        slice_x.append(xx)
        # print slice_x
        return np.array(slice_x)

    def train(self):
        # for ii in range(0, 50):
        output = []
        mp = self.x_filters
        activation_c = []
        pooling = []
        # Check for output layer
        out_layer = len(self.sizes[1:]) - 1
        # Goes through each layer
        outs = []
        for k, (j, fil) in enumerate(zip(self.sizes[1:], self.filters)):
            activation = []
            pool = []
            # for f in fil:
            #     print
            #     # Between hidden and output
            #     # fully connected layers
            out_pool = []
            if k == out_layer:
                c = self.convolution_layer(mp, fil)
                self.out_dot.append(c)
                ac = self.softplus(c)
                # self.out_ac.append(ac)
                po = self.max_pooling(ac)
                # print po
                activation.append(ac)
                pool.append(po)
                out_pool.append(0)

                #### next layer 2
                new_f = self.new_f.reshape(self.new_f.shape[1:])
                c = self.convolution_layer(ac, new_f.transpose())
                self.out_dot.append(c)
                ac = self.softplus(c)
                self.act_outputs.append(ac)
                # out = np.max(ac)
                # outs.append(ac)
            # Between input and output
            else:
                # op = self.not_fully_conected(mp)
                # print "here"
                # print mp
                c = self.convolution_layer(mp, fil)
                self.hidden_dot = c
                ac = self.softplus(c)
                # print ac
                self.hidden_ac = ac
                po = self.max_pooling(ac)
                activation.append(ac)
                mp = po
            activation_c.append(activation)
            pooling.append(pool)
        prediction = self.evaluate()
        return prediction

    def evaluate(self):
        pred = []
        # fil_pred = []
        yy = np.array(self.act_outputs)
        yy = yy.flatten()
        if yy[0] > yy[1]:
            pred.append([1, 0])
        else:
            pred.append([0, 1])
        self.act_outputs = []
        fil = random.random()
        if fil < 0.3:
            pred =[[pred[0][1], pred[0][0]]]
        return pred

    def learning(self):
        pass

    def convolution_layer(self, x, f, *args):
        # print x.shape
        # print f.shape
        c = np.dot(x, f)
        # print c
        # print c
        return c

    def max_pooling(self, p, *args):
        mp = []
        z = np.array(p)
        # print "zshape"
        # print z.shape
        if args:
            for ar in args:
                mp = [np.max(i) for i in p]
        else:
            for pp in p:

                xp = [pp[i:i + self.pool_step] for i in range(0, len(pp), self.pool_step)]
                mps = [np.max(i) for i in xp]
                mp.append(mps)
        # mp = [np.max(i) for i in p]
        return mp

    def relu(self, c):
        # print "ReLU"
        # Replace all that less than 0
        # with 0
        c[c < 0] = 0
        return c

    def softmax(self, s):
        soft_exp = np.exp(s)
        soft_sum = np.sum(soft_exp)
        return soft_exp / soft_sum

    def softmax_prime(self, soft):
        sfmax = self.softmax(soft)
        return sfmax * (1.0 - sfmax)

    def softplus(self, s):
        return np.log(1.0 + np.exp(s))

    def sigmoid(self, z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z) * (1.0 - self.sigmoid(z))

    def loss_function(self, yp, y):
        loss = []
        yp = np.array(yp)
        y = np.array(y)
        loss.append(yp - y)
        return loss
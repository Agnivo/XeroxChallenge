import lasagne as L
import theano as TH
import theano.tensor as T
import theano.printing as PP
import numpy as np


def batch_iterable(
    x, y, batch_size
):
    size = x.shape[0]
    num_batches = size / batch_size
    for i in xrange(num_batches):
        yield (x[i * batch_size:(i + 1) * batch_size],
               y[i * batch_size:(i + 1) * batch_size])

    if size % batch_size != 0:
        yield (x[num_batches * batch_size:],
               y[num_batches * batch_size:])


class scaled_softmax:

    def __init__(self, temp):
        self.t = temp

    def __call__(self, x):
        return T.softmax(x * self.t)


class nnet:

    def __init__(
        self, n_in, n_out, h_layers,
        i_drop=None,
        l_drops=None,
        lam=0,
        nonlinearity=L.nonlinearities.sigmoid
    ):

        self.input = T.fmatrix('input')
        self.layers = []

        l_in = L.layers.InputLayer(shape=(None, n_in), input_var=self.input)

        if i_drop is not None:
            curr = L.layers.DropoutLayer(l_in, p=i_drop, rescale=True)
        else:
            curr = l_in

        for i, j in enumerate(h_layers):
            curr = L.layers.DenseLayer(
                curr, num_units=j,
                nonlinearity=nonlinearity,
                W=L.init.GlorotUniform(
                    gain=(1 if nonlinearity == L.nonlinearities.sigmoid
                          else 'relu')
                ),
                b=L.init.Constant(0.0)
            )
            self.layers.append(curr)
            if l_drops is not None and l_drops[i] is not None:
                curr = L.layers.DropoutLayer(
                    curr, p=l_drops[i], rescale=True
                )

        final_nonlinearity = L.nonlinearities.sigmoid

        self.output_layer = L.layers.DenseLayer(
            curr, num_units=n_out,
            nonlinearity=final_nonlinearity
        )
        self.layers.append(self.output_layer)

        self.output = L.layers.get_output(self.output_layer)
        self.test_output = L.layers.get_output(
            self.output_layer, deterministic=True
        )
        self.target = T.fmatrix('target')

        regs = L.layers.get_all_params(self.output_layer, regularizable=True)
        self.reg = T.sum(regs[0] * regs[0])
        for par in regs[1:]:
            self.reg += T.sum(par * par)

        self.loss = L.objectives.categorical_crossentropy(
            self.output, self.target
        )

        self.loss = T.mean(self.loss) + lam * self.reg

    def savemodel(self, filename):
        vals = L.layers.get_all_param_values(self.output_layer)
        np.savez(filename, vals)

    def loadmodel(self, filename):
        vals = np.load(filename)['arr_0']
        L.layers.set_all_param_values(self.output_layer, vals)

    def train(
        self, x, y, lrate, gamma, batch_size, iters,
        test_batch=None,
        testx=None, testy=None,
        filename='model.npz',
        lrate_iters=None, lrate_factor=None,
    ):
        print "Training ... "
        outputs = [self.loss]
        inputs = [self.input, self.target]

        params = L.layers.get_all_params(self.output_layer, trainable=True)
        updates = L.updates.nesterov_momentum(
            self.loss, params, learning_rate=lrate, momentum=gamma)

        self.trainer = TH.function(
            inputs=inputs, outputs=outputs, updates=updates)

        acc = 0.0
        for i in xrange(iters):
            tot_loss = 0.0
            cnt = 0
            for bx, by in batch_iterable(x, y, batch_size):
                c_loss, = self.trainer(bx, by)
                tot_loss += c_loss
                cnt += 1
            print "Iteration {0}, Loss = {1}".format(i, tot_loss / cnt)

            if testx is None or testy is None:
                testx = x
                testy = y

            if i % 10 == 0:
                c_acc = self.test(testx, testy, batch_size)
                if acc < c_acc:
                    self.savemodel(filename)

    def test(
        self, x, y, batch_size
    ):
        print 'Testing ...'

        self.tester = TH.function(
            inputs=[self.input],
            outputs=[self.test_output],
            updates=None
        )

        acc = 0.0
        cnt = 0
        for bx, by in batch_iterable(x, y, batch_size):
            c_out, = self.tester(bx)
            c_acc = np.mean(c_out[0] == by)
            acc += c_acc
            cnt += 1

        acc /= cnt
        print 'Mean accuracy = {0}'.format(acc)
        return acc

    def getclass(self, x):
        self.tester = TH.function(
            inputs=[self.input],
            outputs=[self.test_output],
            updates=None
        )

        return np.argmax(self.tester(x.reshape(1, -1))[0][0])

    def setx(self, xval):
        self.input.set_value(xval)

    def getx(self):
        return self.input.get_value()

    def max_inp(self, layer_num, node_num, l_rate, gamma, iters):
        node = L.layers.get_output(
            self.layers[layer_num], deterministic=True
        )[node_num]

        updates = L.updates.nesterov_momentum(
            node, [self.input], learning_rate=l_rate, momentum=gamma
        )

        maxer = TH.function(
            inputs=[],
            outputs=[node],
            updates=updates
        )

        print 'Maximizing ...'
        for i in xrange(iters):
            cval = maxer()

        print 'Node value =', cval
        return cval

import cPickle
import gzip
import time

import numpy as np
import theano.tensor as T

import climin.stops
import climin.initialize

from brummlearn.mlp import Mlp, DropoutMlp, dropout_optimizer_conf
from brummlearn.data import one_hot

datafile = 'mnist.pkl'
# Load data.

train_set, val_set, test_set = cPickle.load(open(datafile))

X, Z = train_set
VX, VZ = val_set
TX, TZ = test_set

Z = one_hot(Z, 10)
VZ = one_hot(VZ, 10)
TZ = one_hot(TZ, 10)

image_dims = 28, 28

max_passes = 100
batch_size = 250
max_iter = max_passes * X.shape[0] / batch_size
n_report = X.shape[0] / batch_size

stop = climin.stops.any_([
    climin.stops.after_n_iterations(max_iter),
    ])

pause = climin.stops.modulo_n_iterations(n_report)

optimizer = 'rmsprop', {'steprate': 0.001, 'momentum': 0.9, 'decay': 0.9, 'step_adapt': 0.01}
#optimizer = dropout_optimizer_conf(steprate_0=1, n_repeats=1)
m = Mlp(784, [512, 512], 10, hidden_transfers=['sigmoid', 'sigmoid'], out_transfer='softmax', loss='nce', optimizer=optimizer, batch_size=batch_size, max_iter=max_iter)
m.parameters.data[...] = np.random.normal(0, 1, m.parameters.data.shape)

'''weight_decay = ((m.parameters.in_to_hidden**2).sum()
                + (m.parameters.hidden_to_hidden_0**2).sum()
                + (m.parameters.hidden_to_out**2).sum())
weight_decay /= m.exprs['inpt'].shape[0]
m.exprs['true_loss'] = m.exprs['loss']
c_wd = 0.00001
m.exprs['loss'] = m.exprs['loss'] + c_wd * weight_decay'''

#f_wd = m.function(['inpt'], c_wd * weight_decay)
n_wrong = 1 - T.eq(T.argmax(m.exprs['output'], axis=1), T.argmax(m.exprs['target'], axis=1)).mean()
f_n_wrong = m.function(['inpt', 'target'], n_wrong)

losses = []
v_losses = []
print 'max iter', max_iter

start = time.time()
# Set up a nice printout.
keys = '#', 'loss', 'val loss', 'seconds', 'wd', 'train emp', 'val emp'
max_len = max(len(i) for i in keys)
header = '\t'.join(i for i in keys)
print header
print '-' * len(header)

f_loss = m.function(['inpt', 'target', 'mask'], ['loss'])

maskX = np.zeros((X.shape[0],10))
maskVX = np.zeros((VX.shape[0],10))

for i, info in enumerate(m.powerfit((X, Z, maskX), (VX, VZ, maskVX), stop, pause)):
    if info['n_iter'] % n_report != 0:
        continue
    passed = time.time() - start
    losses.append(info['loss'])
    v_losses.append(info['val_loss'])

    #img = tile_raster_images(fe.parameters['in_to_hidden'].T, image_dims, feature_dims, (1, 1))
    #save_and_display(img, 'filters-%i.png' % i)
    info.update({
        'time': passed,
        'train_emp': f_n_wrong(X, Z),
        'val_emp': f_n_wrong(VX, VZ),
    })
    row = '%(n_iter)i\t%(loss)g\t%(val_loss)g\t%(time)g\t%(train_emp)g\t%(val_emp)g' % info
    print row
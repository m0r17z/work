__author__ = 'mo'

import cPickle
import gzip
import time

import numpy as np
import theano.tensor as T

import climin.stops
import climin.initialize
import climin

from breze.learn.autoencoder import AutoEncoder, ContractiveAutoEncoder
import h5py
import warnings
from theano import function
import sklearn
import gc
# Load data.

class Steprates:

	def __init__(self, initial_steprate, annealing_factor, epoch, min_step):
		self.step = initial_steprate
		self.anneal = annealing_factor
		self.epoch = epoch
		self.count = 0
		self.min_step = min_step

	def __iter__(self):
		return self

	def next(self):
		if self.count % self.epoch == 0 and not self.count == 0:
			self.step *= self.anneal
			self.count += 1
			#print 'epoch'
			#print "step: ", self.step
			if self.step < self.min_step:
				return self.min_step
			return self.step
		else:
			#print "step: ", self.step
			self.count += 1
			if self.step < self.min_step:
				return self.min_step
			return self.step

def simple_ae(init_step, anneal, b_size, arch, activs):

	X, mask_X = cPickle.load(open('hand_data_train.pkl'))
	VX, mask_VX = cPickle.load(open('hand_data_val.pkl'))

	X, mask_X = X[:288000], mask_X[:288000]
	VX, mask_VX = VX[:96000], mask_VX[:96000]


	##################### Preprocessing done ##################################
	max_passes = 200
	batch_size = b_size
	max_iter = max_passes * X.shape[0] / batch_size
	n_report = X.shape[0] / batch_size

	stop = climin.stops.any_([
		climin.stops.patience('val_loss',200,1.2, threshold=1e-3),
	    #climin.stops.after_n_iterations(max_iter)
		])

	pause = climin.stops.modulo_n_iterations(n_report)

	steps = Steprates(init_step,anneal,n_report,0.00001)


	optimizer = 'gd', {'steprate': steps, 'momentum': 0.1}
	#optimizer = dropout_optimizer_conf(steprate_0=1, n_repeats=1)
	m = ContractiveAutoEncoder(54, arch, optimizer=optimizer, batch_size=batch_size, hidden_transfers=activs, tied_weights=False,
	                                     out_transfer='identity', loss='squared')
	m.parameters.data[...] = np.random.normal(0, 0.1, m.parameters.data.shape)
	#print m.n_feature
	#print m.feature_transfers

	losses = []
	v_losses = []
	print 'max iter', max_iter

	start = time.time()
	# Set up a nice printout.
	keys = '#', 'loss', 'seconds'
	max_len = max(len(i) for i in keys)
	header = '\t'.join(i for i in keys)
	print header
	print '-' * len(header)

	for i, info in enumerate(m.powerfit((X, mask_X),(VX, mask_VX), stop, pause)):
		if info['n_iter'] % n_report != 0:
			continue
		passed = time.time() - start
		losses.append(info['loss'])

		info.update({
		'time': passed
		})
		row = '%(n_iter)i\t%(loss)g\t%(val_loss)g\t%(time)g' % info
		print row

	return info['best_loss'],info['best_pars']

########### a simple ae implemented with the mlp class and e.g. 54,30,10,30,54
# or 54,10,54 layers gave results of about 1.3-1.4 as best error

steps = [0.01]
anneals = [0.997, 0.9997, 0.99997]
batches = [50,100,200,500]
layers = [[[40,10,40],['sigmoid','identity','sigmoid']],[[30,10,30],['sigmoid','identity','sigmoid']],[[20,10,20],['sigmoid','identity','sigmoid']]
          ,[[100,10,100],['sigmoid','identity','sigmoid']],[[40,20,10,20,40],['sigmoid','sigmoid','identity','sigmoid','sigmoid']]
	,[[30,40,10,40,30],['sigmoid','sigmoid','identity','sigmoid','sigmoid']],[[200,100,10,100,200],['sigmoid','sigmoid','identity','sigmoid','sigmoid']]]
best_loss = np.inf
best_pars = 0
best_arch = 0

'''for arch in layers:
	for i in np.arange(3):
		s_ind = int(np.random.random()*len(steps))
		init_step = steps[s_ind]
		a_ind = int(np.random.random()*len(anneals))
		annealing = anneals[a_ind]
		b_ind = int(np.random.random()*len(batches))
		b_size = batches[b_ind]
		print 'chose initial step of % f, annealing of %f and batchsize %i' %(steps[s_ind], anneals[a_ind], batches[b_ind])
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			best_l, best_p = simple_ae(init_step,annealing,b_size,arch[0],arch[1])
		if best_l < best_loss:
			best_loss = best_l
			best_pars = best_p
			best_arch = arch
			print 'found new best params with loss of %f' %best_loss
			print 'architecture: ' +str(arch)

print '---------------------------------'
print 'best loss is %f' %best_loss
print 'with architecture: ' +str(best_arch)
print '----------------------------------'
best = open('best_autoencoder.pkl','w')
cPickle.dump(best_pars,best)
best.close()'''
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	best_l, best_p = simple_ae(0.01,0.9997,100,[100,10,100],['sigmoid','identity','sigmoid'])
__author__ = 'mo'

import numpy as np
from breze.learn import pca
import matplotlib.pyplot as plt

X = np.loadtxt('hand_data_train_100Hz.txt')
errors = []
f_results = open('pca_results.txt','w')

for nr_eigens in np.arange(1,55):
	principals = pca.Pca(n_components=nr_eigens)
	principals.fit(X)
	X_r = principals.reconstruct(X)
	avg_squared_dist = np.sum((X - X_r)**2,axis=0)/len(X)
	avg_overall = np.sum(avg_squared_dist)/len(avg_squared_dist)
	errors.append(avg_overall)
	print 'for %i eigenvalues the average squared error per feature is %s.\n' % (nr_eigens,str(avg_squared_dist))
	print 'the average squared error over the average per feature is %s.\n' % avg_overall
	print '-------------------------------------------'
	f_results.write('for %i eigenvalues the average squared error per feature is %s.\n' % (nr_eigens,str(avg_squared_dist)))
	f_results.write('the average squared error over the average per feature is %s.\n' % avg_overall)
	f_results.write('-------------------------------------------')

f_results.close()

plot = plt.plot(np.arange(1,55), np.array(errors), 'r-', linewidth=1, label = 'avg overall squared error')
plt.xlabel('# eigenvalues')
plt.ylabel('overall average average error')

plt.legend(shadow=True)

plt.show()
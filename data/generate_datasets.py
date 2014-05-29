__author__ = 'mo'
import h5py
import sklearn.preprocessing as pp
import numpy as np
import cPickle

def seperate_corrupt():
	file = '/home/mo/BRML_work/hand_data/hand_models_100Hz_'
	good_X = []
	#corrupt_X = []
	#good_maskX = []
	#corrupt_maskX = []

	for part in ['1_30.hdf5','31_60.hdf5','61_90.hdf5','91_110.hdf5']:
		print 'going through file ' + file + part
		data = h5py.File(file + part)
		maskX = data['masks']['masks'][...]
		X = data['models']['hand_models'][...]
		print 'found ' +str(len(X)) +' samples'


		################## Preprocessing of the data ##########################
		# invert the mask so that all positions with nan are 0
		maskX += 1
		maskX = np.where(maskX == 2,0,maskX)

		# set nans to zero
		X = np.where(np.isnan(X),0,X)

		################## Sorting out the corrupted data #####################
		for i in np.arange(len(maskX)):
			#print i
			if np.sum(maskX[i]) == 54:
				#print 'sample %i is corrupted with sum %i' % (i,np.sum(maskX[i]))
				#corrupt_X.append(X[i])
				#corrupt_maskX.append(maskX[i])
				good_X.append(X[i])
				#good_maskX.append(maskX[i])

		data.close()
		del maskX
		del X
		#corrupt_X, corrupt_maskX = np.array(corrupt_X), np.array(corrupt_maskX)

		#dump_file = open('corrupt_hand_data.pkl','w')
		#cPickle.dump((corrupt_X,corrupt_maskX), dump_file)
		#dump_file.close()

	print 'found ' +str(len(good_X)) +' good samples'

	#good_X, good_maskX = np.array(good_X), np.array(good_maskX)
	#good_X = np.array(good_X)
	dump_file = open('good_hand_data_100Hz.pkl','w')
	cPickle.dump(good_X, dump_file)
	dump_file.close()

def scale_data():
	good_X = cPickle.load(open('good_hand_data_100Hz.pkl'))
	#corrupt_X, corrupt_maskX = cPickle.load(open('corrupt_hand_data.pkl'))

	scaler = pp.StandardScaler()
	scaler.fit(good_X)
	good_X = scaler.transform(good_X)

	#scaler = pp.StandardScaler()
	#scaler.fit(corrupt_X)
	#corrupt_X = scaler.transform(corrupt_X)

	#corrupt_X *= corrupt_maskX

	#dump_file = open('corrupt_hand_data.pkl','w')
	#cPickle.dump((corrupt_X,corrupt_maskX), dump_file)
	#dump_file.close()

	#dump_file = open('good_hand_data_100Hz_scaled.pkl','w')
	np.savetxt('good_hand_data_100Hz_scaled_frst.txt', good_X[:1000000]) #cPickle.dump(good_X, dump_file)
	np.savetxt('good_hand_data_100Hz_scaled_scnd.txt', good_X[1000000:2000000])
	np.savetxt('good_hand_data_100Hz_scaled_thrd.txt', good_X[2000000:len(good_X)])
	#dump_file.close()

def generate_subsets():

	for suf in ['frst.txt','scnd.txt','thrd.txt']:

		X = np.loadtxt('good_hand_data_100Hz_scaled_'+ suf)

		# split the data into training, validation and test set
		X_list = np.array(X).tolist()
		del X
		#mask_list = np.array(maskX).tolist()

		valX = []
		#val_mask = []
		testX = []
		#test_mask = []

		len_X = len(X_list)
		print 'length of dataset is %i' % len_X
		len_val = 0
		len_test = 0
		while len_val/float(len_X) < 0.2:
			ind = int(np.random.random() * len(X_list))
			valX.append(X_list.pop(ind))
			#val_mask.append(mask_list.pop(ind))
			len_val += 1

		while len_test/float(len_X) < 0.2:
			ind = int(np.random.random() * len(X_list))
			testX.append(X_list.pop(ind))
			#test_mask.append(mask_list.pop(ind))
			len_test += 1

		print len(X_list)
		#print len(mask_list)
		print len(valX)
		#print len(val_mask)
		print len(testX)
		#print len(test_mask)

		#del X
		#del maskX

		trainX = np.array(X_list)
		np.savetxt('hand_data_train_100Hz'+suf,trainX)
		#train_mask = np.array(mask_list)
		#dump_file = open('hand_data_train_100Hz.pkl','w')
		#cPickle.dump((trainX,train_mask), dump_file)
		#dump_file.close()
		del trainX
		#del train_mask
		print 'pickling training set done.'

		valX = np.array(valX)
		np.savetxt('hand_data_val_100Hz'+suf,valX)
		#val_mask = np.array(val_mask)
		#dump_file = open('hand_data_val.pkl','w')
		#Pickle.dump((valX,val_mask), dump_file)
	    #dump_file.close()
		del valX
	    #del val_mask
		print 'pickling validation set done.'

		testX = np.array(testX)
		np.savetxt('hand_data_test_100Hz'+suf,testX)
		#test_mask = np.array(test_mask)
		#dump_file = open('hand_data_test.pkl','w')
	    #cPickle.dump((testX,test_mask), dump_file)
	    #dump_file.close()
		del testX
	    #del test_mask
		print 'pickling test set done.'
		del X_list

def merge_sets():
	X0 = np.loadtxt('hand_data_train_100Hzfrst.txt')
	X1 = np.loadtxt('hand_data_train_100Hzscnd.txt')
	X2 = np.loadtxt('hand_data_train_100Hzthrd.txt')
	X = np.concatenate((X0,X1,X2),axis=0)
	print X.shape
	np.savetxt('hand_data_train_100Hz.txt',X)
	del X0, X1, X2, X

	X0 = np.loadtxt('hand_data_val_100Hzfrst.txt')
	X1 = np.loadtxt('hand_data_val_100Hzscnd.txt')
	X2 = np.loadtxt('hand_data_val_100Hzthrd.txt')
	X = np.concatenate((X0,X1,X2),axis=0)
	print X.shape
	np.savetxt('hand_data_val_100Hz.txt',X)
	del X0, X1, X2, X

	X0 = np.loadtxt('hand_data_test_100Hzfrst.txt')
	X1 = np.loadtxt('hand_data_test_100Hzscnd.txt')
	X2 = np.loadtxt('hand_data_test_100Hzthrd.txt')
	X = np.concatenate((X0,X1,X2),axis=0)
	print X.shape
	np.savetxt('hand_data_test_100Hz.txt',X)
	del X0, X1, X2, X


if __name__ == '__main__':
	merge_sets()

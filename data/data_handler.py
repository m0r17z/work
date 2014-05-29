__author__ = 'mo'

import scipy.io as sp
import numpy as np
import os
import h5py
import matplotlib.pyplot as pl


class DataHandler:
	def __init__(self):
		self.data = []
		self.ref_models = []
		self.ref_masks = []
		self.meas_models = []
		self.meas_masks = []
		self.models = []
		self.masks = []

	def generate_hdf5(self, path):
		print 'starting to load the data.'
		self.load_raw_data(path)
		print 'data has been loaded.'

		print 'generating hand models.'
		#self.ref_models, self.ref_masks = self.compute_reference_models()

		#self.ref_models = np.array(self.ref_models)
		#self.ref_masks = np.array(self.ref_masks)

		self.meas_models, self.meas_masks = self.compute_measure_models()
		self.meas_models = np.array(self.meas_models)
		self.meas_masks = np.array(self.meas_masks)

		self.models = self.meas_models #np.concatenate([self.ref_models, self.meas_models], axis=0)
		self.masks = self.meas_masks #np.concatenate([self.ref_masks, self.meas_masks], axis=0)
		print 'generating hand models done.'
		print 'storing models as hdf5 file.'
		f = h5py.File('/home/mo/BRML_work/hand_data/hand_models_100Hz_test.hdf5', 'w')
		mod_group = f.create_group('models')
		mask_group = f.create_group('masks')
		mod_group.create_dataset('hand_models', self.models.shape, data=self.models)
		mask_group.create_dataset('masks', self.masks.shape, data=self.masks)
		f.close()
		print 'storing models done.'


	def compute_means(self, path):
		print 'computing statistics.'
		file = path + 'hand_models.hdf5'
		data = h5py.File(file)
		masks = data['masks']['masks'][...]
		stats = np.sum(masks,axis=0)/float(len(masks))
		f = open('/home/mo/BRML_work/hand_data/corruption_statistics_100Hz','w')
		for i in np.arange(0,len(stats)):
			f.write('Feature %i: %f\n' %(i,stats[i]))
		f.close()
		print 'computing statistics done.'


	def show_covariance(self, path):
		print 'computing covariance.'
		file = path + 'hand_models.hdf5'
		data = h5py.File(file)
		masks = np.array(data['masks']['masks'][...])
		cov = np.array(np.cov(masks.T), dtype='float32')
		print cov
		pl.gray()
		pl.imshow(cov,interpolation='none')
		pl.show()


	def load_raw_data(self, path):
		for i in np.arange(1, 11):
			nr = str(i)
			if len(nr) == 1:
				nr = '00' + nr
			elif len(nr) == 2:
				nr = '0' + nr
			file_l_path = path + 'proband%sl_all_samples.mat' % nr
			file_r_path = path + 'proband%sr_all_samples.mat' % nr
			if os.path.exists(file_l_path):
				self.data.append(sp.loadmat(file_l_path))
				print 'loaded ' + file_l_path +'.'
			else:
				print '\'' + file_l_path + '\' doesn\'t exist.'
			if os.path.exists(file_r_path):
				self.data.append(sp.loadmat(file_r_path))
				print 'loaded ' + file_r_path +'.'
			else:
				print '\'' + file_r_path + '\' doesn\'t exist.'


	'''def compute_reference_models(self):
		reference_positions = []
		reference_masks = []
		origin = np.zeros((4,))
		origin[3] = 1.0
		corrupt = 0
		for proband in np.arange(0, len(self.data)):
			reference_pos = np.zeros((54,))
			reference_mas = np.zeros((54,))
			for m_star in np.arange(0, len(self.data[proband]['T_ref0'][0][0])):
				t_matrix = np.zeros((4, 4))
				for row in np.arange(0, 4):
					if corrupt == 1:
						break
					for col in np.arange(0, 4):
						if np.isnan(self.data[proband]['T_ref0'][row][col][m_star]):
							corrupt = 1
							break
						t_matrix[row][col] = self.data[proband]['T_ref0'][row][col][m_star]
				if corrupt == 1:
					reference_pos[m_star * 3 + 0] = np.nan
					reference_pos[m_star * 3 + 1] = np.nan
					reference_pos[m_star * 3 + 2] = np.nan
					reference_mas[m_star * 3 + 0] = 1
					reference_mas[m_star * 3 + 1] = 1
					reference_mas[m_star * 3 + 2] = 1
					corrupt = 0
				else:
					position = np.dot(t_matrix, origin)
					reference_pos[m_star * 3 + 0] = position[0]
					reference_pos[m_star * 3 + 1] = position[1]
					reference_pos[m_star * 3 + 2] = position[2]
			for marker in np.arange(0, len(self.data[proband]['T_ref0_pos'][0])):
				for row in np.arange(0, 3):
					reference_pos[(13 + marker) * 3 + row] = self.data[proband]['T_ref0_pos'][row][marker]
					if np.isnan(self.data[proband]['T_ref0_pos'][row][marker]):
						reference_mas[(13 + marker) * 3 + row] = 1
			reference_positions.append(reference_pos)
			reference_masks.append(reference_mas)

		return reference_positions, reference_masks'''

	def compute_measure_models(self):
		measure_positions = []
		measure_masks = []
		corrupt = 0
		for proband in np.arange(0, len(self.data)):
			for sample in np.arange(0, len(self.data[proband]['T_meas'][0][0][0])):
				measure_pos = np.zeros((54,))
				measure_mas = np.zeros((54,))
				for m_star in np.arange(0, len(self.data[proband]['T_meas'][0][0])):
					t_matrix = np.zeros((4, 4))
					for row in np.arange(0, 4):
						if corrupt == 1:
							break
						for col in np.arange(0, 4):
							if np.isnan(self.data[proband]['T_meas'][row][col][m_star][sample]):
								corrupt = 1
								break
							t_matrix[row][col] = self.data[proband]['T_meas'][row][col][m_star][sample]
					if corrupt == 1:
						measure_pos[m_star * 3 + 0] = np.nan
						measure_pos[m_star * 3 + 1] = np.nan
						measure_pos[m_star * 3 + 2] = np.nan
						measure_mas[m_star * 3 + 0] = 1
						measure_mas[m_star * 3 + 1] = 1
						measure_mas[m_star * 3 + 2] = 1
						corrupt = 0
					else:
						measure_pos[m_star * 3 + 0] = t_matrix[0][3]
						measure_pos[m_star * 3 + 1] = t_matrix[1][3]
						measure_pos[m_star * 3 + 2] = t_matrix[2][3]
				for marker in np.arange(0, len(self.data[proband]['T_meas_pos'][0])):
					for row in np.arange(0, 3):
						measure_pos[(13 + marker) * 3 + row] = self.data[proband]['T_meas_pos'][row][marker][sample]
						if np.isnan(self.data[proband]['T_meas_pos'][row][marker][sample]):
							measure_mas[(13 + marker) * 3 + row] = 1
				measure_positions.append(measure_pos)
				if np.sum(measure_mas) == 0:
					print 'found good model: '
					print measure_pos
				measure_masks.append(measure_mas)

		return measure_positions, measure_masks


	def check_nan(self):
		result = open('/home/mo/BRML_work/hand_data/nan_result_100Hz', 'w')
		'''found_nan_pos = 0
		nr_bad_pos = 0
		nr_pos = 0
		found_nan_mats = 0
		nr_bad_mats = 0
		nr_mats = 0
		found_bad_ref = 0
		nr_bad_refs = 0
		nr_refs = 0
		for i in np.arange(0, len(self.data)):
			for j in np.arange(0, len(self.data[i]['T_meas_pos'][0][0])):
				for l in np.arange(0, 3):
					if np.isnan(self.data[i]['T_meas_pos'][l][j]):
						found_nan_pos = 1
						found_bad_ref = 1

						break
				if found_nan_pos == 1:
					nr_bad_pos += 1
					found_nan_pos = 0
				nr_pos += 1

			for j in np.arange(0, len(self.data[i]['T_meas'][0][0])):
				for l in np.arange(0, 4):
					if found_nan_mats == 1:
						break
					for m in np.arange(0, 4):
						if np.isnan(self.data[i]['T_meas'][l][m][j]):
							found_nan_mats = 1
							found_bad_ref = 1
							break
				if found_nan_mats == 1:
					nr_bad_mats += 1
					found_nan_mats = 0
				nr_mats += 1
			if found_bad_ref == 1:
				nr_bad_refs += 1
				found_bad_ref = 0
			nr_refs += 1
		print 'found %i positions containing a NaN value in \'T_ref0_pos\'.' % nr_bad_pos
		print 'that is %f percent of all positions.\n' % (float(nr_bad_pos) / float(nr_pos) * 100)
		print 'found %i matrices containing a NaN value in \'T_ref0\'.' % nr_bad_mats
		print 'that is %f percent of all matrices.\n' % (float(nr_bad_mats) / float(nr_mats) * 100)
		print 'found %i reference hand models with at least one missing position.' % nr_bad_refs
		print 'that is %f percent of all reference hand models.\n' % (float(nr_bad_refs) / float(nr_refs) * 100)
		result.write('found %i positions containing a NaN value in \'T_ref0_pos\'.\n' % nr_bad_pos)
		result.write('that is %f percent of all positions.\n\n' % (float(nr_bad_pos) / float(nr_pos) * 100))
		result.write('found %i matrices containing a NaN value in \'T_ref0\'.\n' % nr_bad_mats)
		result.write('that is %f percent of all matrices.\n\n' % (float(nr_bad_mats) / float(nr_mats) * 100))
		result.write('found %i reference hand models with at least one missing position.\n' % nr_bad_refs)
		result.write(
			'that is %f percent of all reference hand models.\n\n' % (float(nr_bad_refs) / float(nr_refs) * 100))'''

		found_nan_pos = 0
		nr_bad_pos = 0
		nr_pos = 0
		found_nan_mat = 0
		nr_bad_mats = 0
		nr_mats = 0
		found_bad_mod = 0
		nr_bad_mod = 0
		nr_mod = 0
		for i in np.arange(0, len(self.data)):
			for j in np.arange(0, len(self.data[i]['T_meas_pos'][0][0])):
				for k in np.arange(0, len(self.data[i]['T_meas_pos'][0])):
					for l in np.arange(0, 3):
						if np.isnan(self.data[i]['T_meas_pos'][l][k][j]):
							found_nan_pos = 1
							found_bad_mod = 1
							break
					if found_nan_pos == 1:
						nr_bad_pos += 1
						found_nan_pos = 0
					nr_pos += 1

				for k in np.arange(0, len(self.data[i]['T_meas'][0][0])):
					for l in np.arange(0, 4):
						if found_nan_mat == 1:
							break
						for m in np.arange(0, 4):
							if np.isnan(self.data[i]['T_meas'][l][m][k][j]):
								found_nan_mat = 1
								found_bad_mod = 1
								break
					if found_nan_mat == 1:
						nr_bad_mats += 1
						found_nan_mat = 0
					nr_mats += 1
				if found_bad_mod == 1:
					found_bad_mod = 0
					nr_bad_mod += 1
				nr_mod += 1

		print 'found %i positions containing a NaN value in \'T_meas_pos\'.' % nr_bad_pos
		print 'that is %f percent of all positions.\n' % (float(nr_bad_pos) / float(nr_pos) * 100)
		print 'found %i matrices containing a NaN value in \'T_meas\' of all the proband data.' % nr_bad_mats
		print 'that is %f percent of all matrices.\n' % (float(nr_bad_mats) / float(nr_mats) * 100)
		print 'found %i hand models with at least one missing position.' % nr_bad_mod
		print 'that is %f percent of all hand models.\n' % (float(nr_bad_mod) / float(nr_mod) * 100)
		result.write('found %i positions containing a NaN value in \'T_meas_pos\'.\n' % nr_bad_pos)
		result.write('that is %f percent of all positions.\n\n' % (float(nr_bad_pos) / float(nr_pos) * 100))
		result.write('found %i matrices containing a NaN value in \'T_meas\' of all the proband data.\n' % nr_bad_mats)
		result.write('that is %f percent of all matrices.\n\n' % (float(nr_bad_mats) / float(nr_mats) * 100))
		result.write('found %i hand models with at least one missing position.\n' % nr_bad_mod)
		result.write('that is %f percent of all hand models.\n\n' % (float(nr_bad_mod) / float(nr_mod) * 100))

		result.close()


handler = DataHandler()
handler.generate_hdf5('/home/mo/BRML_work/hand_data_raw_100Hz/data/')
#handler.load_raw_data('/home/mo/BRML_work/hand_data_raw_100Hz/data/')
#handler.compute_measure_models()
#handler.generate_hdf5('/home/mo/BRML_work/hand_data_raw/')
#handler.show_covariance('/home/mo/BRML_work/hand_data/')

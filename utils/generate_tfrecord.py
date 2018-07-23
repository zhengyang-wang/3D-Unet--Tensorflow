import tensorflow as tf
import nibabel as nib
import numpy as np
import os


"""Generate TFRecord Files.
"""

################################################################################
# Arguments
################################################################################
RAW_DATA_DIR = '/tempspace2/zwang6/InfantBrain/RawData'
OUTPUT_DIR = '/tempspace2/zwang6/InfantBrain/tfrecords'
VALID_ID = 9
OVERLAP_STEPSIZE = 8
_D = 32
_H = 32
_W = 32


################################################################################
# Functions
################################################################################
def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def cut_edge(data):
	'''Cuts zero edge for a 3D image.

	Args:
		data: A 3D image, [Depth, Height, Width, 1].

	Returns:
		A list of six integers [Depth_s, Depth_e, Height_s, Height_e, Width_s, Width_e]
	'''

	D, H, W, _ = data.shape
	D_s, D_e = 0, D-1
	H_s, H_e = 0, H-1
	W_s, W_e = 0, W-1

	while D_s < D:
		if data[D_s].sum() != 0:
			break
		D_s += 1
	while D_e > D_s:
		if data[D_e].sum() != 0:
			break
		D_e -= 1
	while H_s < H:
		if data[:,H_s].sum() != 0:
			break
		H_s += 1
	while H_e > H_s:
		if data[:,H_e].sum() != 0:
			break
		H_e -= 1
	while W_s < W:
		if data[:,:,W_s].sum() != 0:
			break
		W_s += 1
	while W_e > W_s:
		if data[:,:,W_e].sum() != 0:
			break
		W_e -= 1
	
	return [int(D_s), int(D_e+1), int(H_s), int(H_e+1), int(W_s), int(W_e+1)]


def convert_labels(labels):
	'''Converts 0:background / 10:CSF / 150:GM / 250:WM to 0/1/2/3. SLOW!
	'''

	D, H, W, C = labels.shape

	for d in range(D):
		for h in range(H):
			for w in range(W):
				if labels[d,h,w,0] == 10:
					labels[d,h,w,0] = 1
				elif labels[d,h,w,0] == 150:
					labels[d,h,w,0] = 2
				elif labels[d,h,w,0] == 250:
					labels[d,h,w,0] = 3


def load_subject(data_dir, subject_id):
	'''Load subject data.

	Args:
		subject_id: [1-10]

	Returns:
		[T1, T2, label]
	'''

	subject_name = 'subject-%d-' % subject_id

	f1 = os.path.join(data_dir, subject_name+'T1.hdr')
	f2 = os.path.join(data_dir, subject_name+'T2.hdr')
	fl = os.path.join(data_dir, subject_name+'label.hdr')

	img_T1 = nib.load(f1)
	img_T2 = nib.load(f2)
	img_label = nib.load(fl)

	inputs_T1 = img_T1.get_data()
	inputs_T2 = img_T2.get_data()
	inputs_label = img_label.get_data()

	return [inputs_T1, inputs_T2, inputs_label]
		

def prepare_validation(cutted_image, overlap_stepsize):
	"""Determine patches for validation."""

	patch_ids = []

	D, H, W, _ = cutted_image.shape

	drange = list(range(0, D-_D+1, overlap_stepsize))
	hrange = list(range(0, H-_H+1, overlap_stepsize))
	wrange = list(range(0, W-_W+1, overlap_stepsize))

	if (D-_D) % overlap_stepsize != 0:
		drange.append(D-_D)
	if (H-_H) % overlap_stepsize != 0:
		hrange.append(H-_H)
	if (W-_W) % overlap_stepsize != 0:
		wrange.append(W-_W)

	for d in drange:
		for h in hrange:
			for w in wrange:
				patch_ids.append((d, h, w))

	return patch_ids


def write_training_examples(T1, T2, label, cut_size, output_file):
	"""Create a training tfrecord file.
	
	Args:
		T1: T1 image. [Depth, Height, Width, 1].
		T2: T2 image. [Depth, Height, Width, 1].
		label: Label. [Depth, Height, Width, 1].
		cut_size: A list of six integers [Depth_s, Depth_e, Height_s, Height_e, Width_s, Width_e].
		output_file: The file name for the tfrecord file.
	"""

	writer = tf.python_io.TFRecordWriter(output_file)

	example = tf.train.Example(features=tf.train.Features(
		feature={
			'T1': _bytes_feature([T1[:,:,:,0].tostring()]), #int16
			'T2': _bytes_feature([T2[:,:,:,0].tostring()]), #int16
			'label': _bytes_feature([label[:,:,:,0].tostring()]), #uint8
			'cut_size': _int64_feature(cut_size)
		}
	))

	writer.write(example.SerializeToString())

	writer.close()


def write_validation_examples(T1, T2, label, cut_size, output_file):
	"""Create a validation tfrecord file.
	
	Args:
		T1: T1 image. [Depth, Height, Width, 1].
		T2: T2 image. [Depth, Height, Width, 1].
		label: Label. [Depth, Height, Width, 1].
		cut_size: A list of six integers [Depth_s, Depth_e, Height_s, Height_e, Width_s, Width_e].
		output_file: The file name for the tfrecord file.
	"""

	T1 = T1[cut_size[0]:cut_size[1], cut_size[2]:cut_size[3], cut_size[4]:cut_size[5], :]
	T2 = T2[cut_size[0]:cut_size[1], cut_size[2]:cut_size[3], cut_size[4]:cut_size[5], :]
	label = label[cut_size[0]:cut_size[1], cut_size[2]:cut_size[3], cut_size[4]:cut_size[5], :]

	patch_ids = prepare_validation(T1, OVERLAP_STEPSIZE)
	print ('Number of patches:', len(patch_ids))

	writer = tf.python_io.TFRecordWriter(output_file)

	for i in range(len(patch_ids)):

		(d, h, w) = patch_ids[i]

		_T1 = T1[d:d+_D, h:h+_H, w:w+_W, :]
		_T2 = T2[d:d+_D, h:h+_H, w:w+_W, :]
		_label = label[d:d+_D, h:h+_H, w:w+_W, :]

		example = tf.train.Example(features=tf.train.Features(
			feature={
				'T1': _bytes_feature([_T1[:,:,:,0].tostring()]), #int16
				'T2': _bytes_feature([_T2[:,:,:,0].tostring()]), #int16
				'label': _bytes_feature([_label[:,:,:,0].tostring()]), #uint8
			}
		))

		writer.write(example.SerializeToString())

	writer.close()


def generate_files(output_path, valid_id):
	"""Create tfrecord files."""

	if not os.path.exists(output_path):
		os.makedirs(output_path)

	for i in range(1, 11):
		print('---Process subject %d:---' % i)
		subject_name = 'subject-%d' % i

		filename = os.path.join(output_path, subject_name+'.tfrecords')
		
		if not os.path.isfile(filename) or i == valid_id:
			print('Loading data...')
			[_T1, _T2, _label] = load_subject(RAW_DATA_DIR, i)

			print('Converting label...')
			convert_labels(_label)
			print('Check label: ', np.max(_label))

			cut_size = cut_edge(_T1)
			print('Check cut_size: ',cut_size)

		if not os.path.isfile(filename):
			print('Create the training file:')
			write_training_examples(_T1, _T2, _label, cut_size, filename)

		if i == valid_id:
			subject_name = 'subject-%d-valid-%s' % (valid_id, OVERLAP_STEPSIZE)
			filename = os.path.join(output_path, subject_name+'.tfrecords')

			if not os.path.isfile(filename):
				print('Create the validation file:')
				write_validation_examples(_T1, _T2, _label, cut_size, filename)

			# save converted label for fast evaluation
			converted_label_filename = 'subject-%d-label.npy' % valid_id
			converted_label_filename = os.path.join(output_path, converted_label_filename)

			if not os.path.isfile(converted_label_filename):
				print('Create the converted label file:')
				np.save(converted_label_filename, _label[:,:,:,0])

		print('---Done.---')


if __name__ == '__main__':
	generate_files(OUTPUT_DIR, VALID_ID)

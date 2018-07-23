import tensorflow as tf
import os


"""This script defines the input interface.
"""


################################################################################
# Arguments
################################################################################
_NUM_DATA_FILES = 10
_IMAGE_SHAPE = (144, 192, 256)
_D = 32
_H = 32
_W = 32
_PATCH_SHAPE = (_D, _H, _W)


################################################################################
# Functions
################################################################################
def get_filenames(data_dir, training, valid_id, overlap_step):
	"""Returns a list of filenames."""
	
	data_dir = os.path.join(data_dir, 'tfrecords')

	assert os.path.exists(data_dir), \
			('Run utils/generate_tfrecord.py to generate TFRecord files.')

	if training:
		return [
			os.path.join(data_dir, 'subject-%d.tfrecords' % i)
			for i in range(1, _NUM_DATA_FILES + 1)
			if i != valid_id
		]
	else:
		valid_file = os.path.join(data_dir, 
			'subject-%d-valid-%d.tfrecords' % (valid_id, overlap_step))
		assert os.path.isfile(valid_file), \
			('Run utils/generate_tfrecord.py to generate validation files.')
		return [valid_file]


def decode_train(serialized_example):
	"""Parses training data from the given `serialized_example`."""

	features = tf.parse_single_example(
					serialized_example,
					features={
						'T1':tf.FixedLenFeature([],tf.string),
						'T2':tf.FixedLenFeature([], tf.string),
						'label':tf.FixedLenFeature([],tf.string),
						'cut_size':tf.FixedLenFeature(6, tf.int64)
					})

	cut_size = features['cut_size']

	# Convert from a scalar string tensor
	image_T1 = tf.decode_raw(features['T1'], tf.int16)
	image_T1 = tf.reshape(image_T1, _IMAGE_SHAPE)
	image_T2 = tf.decode_raw(features['T2'], tf.int16)
	image_T2 = tf.reshape(image_T2, _IMAGE_SHAPE)
	label = tf.decode_raw(features['label'], tf.uint8)
	label = tf.reshape(label, _IMAGE_SHAPE)

	# Convert dtype.
	image_T1 = tf.cast(image_T1, tf.float32)
	image_T2 = tf.cast(image_T2, tf.float32)
	label = tf.cast(label, tf.float32)

	return image_T1, image_T2, label, cut_size


def decode_valid(serialized_example):
	"""Parses validation data from the given `serialized_example`."""

	features = tf.parse_single_example(
					serialized_example,
					features={
						'T1':tf.FixedLenFeature([],tf.string),
						'T2':tf.FixedLenFeature([], tf.string),
						'label':tf.FixedLenFeature([],tf.string)
					})

	# Convert from a scalar string tensor
	image_T1 = tf.decode_raw(features['T1'], tf.int16)
	image_T1 = tf.reshape(image_T1, _PATCH_SHAPE)
	image_T2 = tf.decode_raw(features['T2'], tf.int16)
	image_T2 = tf.reshape(image_T2, _PATCH_SHAPE)
	label = tf.decode_raw(features['label'], tf.uint8)
	label = tf.reshape(label, _PATCH_SHAPE)

	# Convert dtype.
	image_T1 = tf.cast(image_T1, tf.float32)
	image_T2 = tf.cast(image_T2, tf.float32)

	return image_T1, image_T2, label


def crop_image(image_T1, image_T2, label, cut_size):
	"""Crop training data."""

	data = tf.stack([image_T1, image_T2, label], axis=-1)

	# Randomly crop a [_D, _H, _W] section of the image.
	image = tf.random_crop(
				data[cut_size[0]:cut_size[1], cut_size[2]:cut_size[3], cut_size[4]:cut_size[5], :],
				[_D, _H, _W, 3])

	[image_T1, image_T2, label] = tf.unstack(image, 3, axis=-1)

	return image_T1, image_T2, label


def normalize_image(image_T1, image_T2, label):
	"""Normalize data."""

	# Subtract off the mean and divide by the variance of the pixels.
	image_T1 = tf.image.per_image_standardization(image_T1)
	image_T2 = tf.image.per_image_standardization(image_T2)

	features = tf.stack([image_T1, image_T2], axis=-1)

	label = tf.cast(label, tf.int32)
	
	return features, label


def input_function(data_dir, training, batch_size, buffer_size, valid_id,
						overlap_step, num_epochs=1, num_parallel_calls=1):
	"""Input function.

	Args:
		data_dir: The directory containing the input data.
		training: A boolean denoting whether the input is for training.
		batch_size: The number of samples per batch.
		buffer_size: The buffer size to use when shuffling records. A larger
			value results in better randomness, but smaller values reduce startup
			time and use less memory.
		valid_id: The ID of the validation subject.
		overlap_step: An integer.
		num_epochs: The number of epochs to repeat the dataset.
		num_parallel_calls: The number of records that are processed in parallel.
			This can be optimized per data set but for generally homogeneous data
			sets, should be approximately the number of available CPU cores.

	Returns:
		Dataset of (features, labels) pairs ready for iteration.
	"""

	with tf.name_scope('input'):
		# Generate a Dataset with raw records.
		filenames = get_filenames(data_dir, training, valid_id, overlap_step)
		dataset = tf.data.TFRecordDataset(filenames)

		# We prefetch a batch at a time, This can help smooth out the time taken to
		# load input files as we go through shuffling and processing.
		dataset = dataset.prefetch(buffer_size=batch_size)

		if training:
			# Shuffle the records. Note that we shuffle before repeating to ensure
			# that the shuffling respects epoch boundaries.
			dataset = dataset.shuffle(buffer_size=buffer_size)

		# If we are training over multiple epochs before evaluating, repeat the
		# dataset for the appropriate number of epochs.
		dataset = dataset.repeat(num_epochs)

		if training:
			dataset = dataset.map(decode_train, num_parallel_calls=num_parallel_calls)
			dataset = dataset.map(crop_image, num_parallel_calls=num_parallel_calls)
		else:
			dataset = dataset.map(decode_valid, num_parallel_calls=num_parallel_calls)

		dataset = dataset.map(normalize_image, num_parallel_calls=num_parallel_calls)

		dataset = dataset.batch(batch_size)

		# Operations between the final prefetch and the get_next call to the iterator
		# will happen synchronously during run time. We prefetch here again to
		# background all of the above processing work and keep it out of the
		# critical training path.
		dataset = dataset.prefetch(1)

		iterator = dataset.make_one_shot_iterator()
		features, label = iterator.get_next()

		return features, label

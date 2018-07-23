import tensorflow as tf


"""This script defines basic operations.
"""



################################################################################
# Basic operations building the network
################################################################################
def Pool3d(inputs, kernel_size, strides):
	"""Performs 3D max pooling."""

	return tf.layers.max_pooling3d(
			inputs=inputs,
			pool_size=kernel_size,
			strides=strides,
			padding='same')


def Deconv3D(inputs, filters, kernel_size, strides, use_bias=False):
	"""Performs 3D deconvolution without bias and activation function."""

	return tf.layers.conv3d_transpose(
			inputs=inputs,
			filters=filters,
			kernel_size=kernel_size,
			strides=strides,
			padding='same',
			use_bias=use_bias,
			kernel_initializer=tf.truncated_normal_initializer())


def Conv3D(inputs, filters, kernel_size, strides, use_bias=False):
	"""Performs 3D convolution without bias and activation function."""

	return tf.layers.conv3d(
			inputs=inputs,
			filters=filters,
			kernel_size=kernel_size,
			strides=strides,
			padding='same',
			use_bias=use_bias,
			kernel_initializer=tf.truncated_normal_initializer())


def Dilated_Conv3D(inputs, filters, kernel_size, dilation_rate, use_bias=False):
	"""Performs 3D dilated convolution without bias and activation function."""

	return tf.layers.conv3d(
			inputs=inputs,
			filters=filters,
			kernel_size=kernel_size,
			strides=1,
			dilation_rate=dilation_rate,
			padding='same',
			use_bias=use_bias,
			kernel_initializer=tf.truncated_normal_initializer())


def BN_ReLU(inputs, training):
	"""Performs a batch normalization followed by a ReLU6."""

	# We set fused=True for a significant performance boost. See
	# https://www.tensorflow.org/performance/performance_guide#common_fused_ops
	inputs = tf.layers.batch_normalization(
				inputs=inputs,
				axis=-1,
				momentum=0.997,
				epsilon=1e-5,
				center=True,
				scale=True,
				training=training, 
				fused=True)

	return tf.nn.relu6(inputs)

import tensorflow as tf


"""This script defines hyperparameters.
"""

def configure():
	flags = tf.app.flags

	# training
	flags.DEFINE_string('raw_data_dir', '/data/zhengyang/InfantBrain/RawData',
			'the directory where the raw data is stored')
	flags.DEFINE_string('data_dir', '/data/zhengyang/InfantBrain/tfrecords_full',
			'the directory where the input data is stored')
	flags.DEFINE_integer('num_training_subs', 9,
			'the number of subjects used for training')
	flags.DEFINE_integer('train_epochs', 100000,
			'the number of epochs to use for training')
	flags.DEFINE_integer('epochs_per_eval', 5000,
			'the number of training epochs to run between evaluations')
	flags.DEFINE_integer('batch_size', 5,
			'the number of examples processed in each training batch')
	flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
	flags.DEFINE_float('weight_decay', 2e-6, 'weight decay rate')
	flags.DEFINE_integer('num_parallel_calls', 5,
			'The number of records that are processed in parallel \
			during input processing. This can be optimized per data set but \
			for generally homogeneous data sets, should be approximately the \
			number of available CPU cores.')
	flags.DEFINE_string('model_dir', './model-10',
			'the directory where the model will be stored')

	# validation / prediction
	flags.DEFINE_integer('patch_size', 32, 'spatial size of patches')
	flags.DEFINE_integer('overlap_step', 8,
			'overlap step size when performing validation/prediction')
	flags.DEFINE_integer('validation_id', 10,
			'1-10 or -1, which subject is used for validation')
	flags.DEFINE_integer('prediction_id', 11,
			'1-23, which subject is used for prediction')
	flags.DEFINE_integer('checkpoint_num', 153000,
			'which checkpoint is used for validation/prediction')
	flags.DEFINE_string('save_dir', './results',
			'the directory where the prediction is stored')

	# network
	flags.DEFINE_integer('network_depth', 3, 'the network depth')
	flags.DEFINE_integer('num_classes', 4, 'the number of classes')
	flags.DEFINE_integer('num_filters', 32,
			 'number of filters for initial_conv')
	
	flags.FLAGS.__dict__['__parsed'] = False
	return flags.FLAGS


conf = configure()